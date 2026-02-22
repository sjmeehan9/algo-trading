"""Live market data streaming via broker adapter.

This module provides the ``LiveData`` class which performs a historical
warmup fetch followed by a real-time bar subscription through any
``BrokerAdapter`` implementation, feeding incoming data into a
``StreamQueue`` for downstream consumption.
"""

import datetime
import logging
import os
from pathlib import Path

from algotrading.src.broker import (
    BarData,
    BrokerAdapter,
    ContractSpec,
    InstrumentType,
    InteractiveBrokersAdapter,
)

from ..exceptions import BrokerConnectionError, DataError
from ..load_config import config_loader
from .stream_queue import StreamQueue

logger = logging.getLogger(__name__)


class LiveData:
    """Stream live market data using a broker adapter.

    Performs an initial historical data warmup, validates temporal
    continuity, then subscribes to real-time bars and routes them
    into a ``StreamQueue`` for downstream processing.

    Args:
        config: Runtime configuration dictionary.
        pipeline: Pipeline configuration dictionary.
        adapter: Optional broker adapter; defaults to
            ``InteractiveBrokersAdapter``.
    """

    CONFIG_FILENAME = "live_streaming.yml"
    HISTORICAL_CONFIG = "historical_data.yml"
    CURRENT_BAR = ""
    INIT_REQUEST_ID = 1000
    DATE_COLUMN = "date"

    def __init__(
        self,
        config: dict,
        pipeline: dict,
        adapter: BrokerAdapter | None = None,
    ):

        self.logger = logger

        self.config = config
        self.pipeline = pipeline
        self.adapter = adapter if adapter is not None else InteractiveBrokersAdapter()
        self.done = False
        self._connected = False
        self._subscription_id: int | None = None

        self.queue = StreamQueue(self.config, self.pipeline)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = Path(current_dir).parents[1]

        config_file_path = os.path.join(parent_dir, "config/", self.CONFIG_FILENAME)
        self.script_config = config_loader(config_file_path, validate=False)

        historical_config_path = os.path.join(
            parent_dir, "config/", self.HISTORICAL_CONFIG
        )
        self.historical_config = config_loader(historical_config_path, validate=False)

        contract_info = self.pipeline["pipeline"]["contract_info"]
        instrument_type_map = {
            "STK": InstrumentType.STOCK,
            "OPT": InstrumentType.OPTION,
            "FUT": InstrumentType.FUTURE,
            "CRYPTO": InstrumentType.CRYPTO,
            "IND": InstrumentType.INDEX,
            "INDEX": InstrumentType.INDEX,
        }
        sec_type = str(contract_info["secType"]).upper()
        if sec_type not in instrument_type_map:
            raise ValueError(f"Unsupported secType for ContractSpec: {sec_type}")

        self.contract_spec = ContractSpec(
            symbol=contract_info["symbol"],
            instrument_type=instrument_type_map[sec_type],
            exchange=contract_info["exchange"],
            currency=contract_info["currency"],
            primary_exchange=contract_info.get("primaryExchange"),
            expiry=contract_info.get("lastTradeDateOrContractMonth")
            or contract_info.get("expiry"),
            strike=(
                float(contract_info["strike"]) if "strike" in contract_info else None
            ),
            right=contract_info.get("right"),
            multiplier=(
                float(contract_info["multiplier"])
                if "multiplier" in contract_info
                and contract_info["multiplier"] is not None
                else 1.0
            ),
        )

        self.live_info = self.pipeline["pipeline"]["live_data_config"]
        self.historical_info = self.pipeline["pipeline"]["historical_data_config"]

        self.bar_columns = self.script_config["bar_columns"]
        self.historical_columns = self.historical_config["bar_columns"]

        self.step_size = self.historical_config["step_size"][
            self.historical_info["barSizeSetting"]
        ]
        self.data_list = []

        self.timer = self.setTimer()

    def connect(self, ip_address: str, port: int, client_id: int) -> None:
        """Establish a broker connection for live data streaming.

        Args:
            ip_address: Broker gateway host address.
            port: Broker gateway port.
            client_id: Unique client identifier for this connection.

        Raises:
            BrokerConnectionError: If the connection attempt fails.
        """
        try:
            self.adapter.connect(ip_address, port, client_id)
            self._connected = True
        except Exception as exc:
            raise BrokerConnectionError(
                "Failed to connect for live streaming",
                broker_name="interactive_brokers",
                host=ip_address,
                port=port,
                context={"client_id": client_id, "error": str(exc)},
            ) from exc

    def run(self) -> None:
        """Start the live data streaming workflow.

        Raises:
            BrokerConnectionError: If called before ``connect``.
        """
        if not self._connected:
            raise BrokerConnectionError(
                "LiveData.run called before broker connection was established",
                broker_name="interactive_brokers",
                host=self.config.get("ip_address"),
                port=self.config.get("port"),
                context={"client_id": self.pipeline["pipeline"]["client_id"]["live"]},
            )
        self.start()

    def disconnect(self) -> None:
        """Disconnect the broker adapter and reset connection state."""
        self.adapter.disconnect()
        self._connected = False

    def setTimer(self) -> int:
        """Return the configured streaming runtime in seconds."""
        runtime = self.pipeline["pipeline"]["live_data_config"]["runtime"]
        return runtime

    def sendRequests(self) -> None:
        """Perform historical warmup and start the real-time subscription."""
        try:
            bars = self.adapter.request_historical_data(
                self.contract_spec,
                datetime.datetime.now(datetime.timezone.utc),
                self.step_size["durationString"],
                self.historical_info["barSizeSetting"],
                self.historical_info["whatToShow"],
            )
        except Exception as exc:
            raise BrokerConnectionError(
                "Failed requesting live historical warmup data",
                broker_name="interactive_brokers",
                host=self.config.get("ip_address"),
                port=self.config.get("port"),
                context={"request_id": self.req_it, "error": str(exc)},
            ) from exc

        self.data_list = [self._bar_to_historical_row(bar) for bar in bars]
        self.CURRENT_BAR = ""

        self.data_validation = self.dataValidation()

        if self.data_validation:
            self.logger.info("Data validation passed")
        else:
            self.logger.warning(
                "Data validation failed, dropping historical warmup data"
            )
            self.data_list = []

        self._subscription_id = self.adapter.subscribe_realtime_data(
            contract=self.contract_spec,
            bar_size=self.live_info["barSizeSetting"],
            data_type=self.live_info["whatToShow"],
            callback=self._on_realtime_bar,
        )

    def _bar_to_historical_row(self, bar: BarData) -> dict[str, object]:
        return {
            self.historical_columns["bar_date"]: int(bar.timestamp.timestamp()),
            self.historical_columns["bar_open"]: bar.open,
            self.historical_columns["bar_high"]: bar.high,
            self.historical_columns["bar_low"]: bar.low,
            self.historical_columns["bar_close"]: bar.close,
            self.historical_columns["bar_volume"]: bar.volume,
            self.historical_columns["bar_wap"]: bar.vwap,
            self.historical_columns["bar_barCount"]: bar.trade_count,
        }

    def _bar_to_live_row(self, bar: BarData) -> dict[str, object]:
        return {
            self.bar_columns["bar_date"]: int(bar.timestamp.timestamp()),
            self.bar_columns["bar_open"]: bar.open,
            self.bar_columns["bar_high"]: bar.high,
            self.bar_columns["bar_low"]: bar.low,
            self.bar_columns["bar_close"]: bar.close,
            self.bar_columns["bar_volume"]: bar.volume,
            self.bar_columns["bar_wap"]: bar.vwap,
            self.bar_columns["bar_barCount"]: bar.trade_count,
        }

    def _on_realtime_bar(self, bar: BarData) -> None:
        time = int(bar.timestamp.timestamp())
        new_row = self._bar_to_live_row(bar)

        self.logger.info("Start of trade data flow: %s", datetime.datetime.now())

        if not self.CURRENT_BAR:
            process = self.connectDates(time)
            if process:
                self.data_list.append(new_row)
                self.queue.put(self.data_list)
                self.CURRENT_BAR = time

        elif self.CURRENT_BAR != time:
            self.queue.put(new_row)

            self.CURRENT_BAR = time

    def dataValidation(self) -> bool:
        """Validate temporal continuity of the historical warmup bars.

        Returns:
            ``True`` if bars span a contiguous time range with the
            expected increment.

        Raises:
            DataError: If no bars were received in the warmup.
        """
        dates = [int(d[self.DATE_COLUMN]) for d in self.data_list]

        if not dates:
            raise DataError(
                "No historical bars received for live data warmup",
                data_source="live_historical_warmup",
                row_count=0,
            )

        self.time_increment = int(self.live_info["barSizeSetting"])

        # Calculate the expected number of entries. Only going to work for seconds
        start_date = dates[0]
        self.end_date = dates[-1]
        expected_count = int((self.end_date - start_date) / self.time_increment) + 1

        # Check if the actual count matches the expected count
        if len(dates) != expected_count:
            return False

        # Check if all expected datetimes are present
        current_date = start_date
        for date in dates:
            if date != current_date:
                return False
            current_date += self.time_increment

        return True

    def connectDates(self, time: int) -> bool:
        """Determine whether the first real-time bar connects to history.

        Args:
            time: Unix timestamp of the incoming real-time bar.

        Returns:
            ``True`` if the bar should be appended and processing
            should begin; ``False`` if data should be discarded.
        """
        connect_time = time - self.time_increment

        if connect_time == self.end_date:
            self.logger.info(
                "Final historical bar increments to real time data, transferring to real time stream"
            )
            return True
        elif connect_time > self.end_date:
            self.logger.info(
                "Gap between historical data end time exists, dropping data"
            )
            self.data_list = []
            return True
        else:
            self.logger.info(
                "Real time data overlaps with historical end time, waiting for new data"
            )
            return False

    def start(self) -> None:
        """Initialize request tracking and begin the streaming flow."""
        self.req_it = self.INIT_REQUEST_ID

        # Request live realTimeBars data
        self.sendRequests()

    def stop(self) -> None:
        """Unsubscribe from real-time data and disconnect the broker."""
        self.logger.info("LiveData connection closed")

        if self._subscription_id is not None:
            self.adapter.unsubscribe_realtime_data(self._subscription_id)
            self._subscription_id = None

        self.done = True
        self.adapter.disconnect()
        self._connected = False
