"""Historical market data collection via broker adapter.

This module provides the ``PastData`` class which iterates over a list of
trading dates and collects intraday OHLCV bars from any ``BrokerAdapter``
implementation, persisting each date's data to a CSV file.
"""

import datetime
import logging
import os
import time
from pathlib import Path

import pandas as pd
from algotrading.src.broker import (
    BarData,
    BrokerAdapter,
    ContractSpec,
    InstrumentType,
    InteractiveBrokersAdapter,
)

from ..exceptions import BrokerConnectionError, DataError
from ..load_config import config_loader

logger = logging.getLogger(__name__)


class PastData:
    """Collect and persist historical market data using a broker adapter.

    Iterates over configured trading dates, requests intraday bars from
    the broker adapter, validates the resulting dataframe, and writes
    each date's data to a CSV file.

    Args:
        config: Runtime configuration dictionary.
        pipeline: Pipeline configuration dictionary.
        adapter: Optional broker adapter; defaults to
            ``InteractiveBrokersAdapter``.
    """

    CONFIG_FILENAME = "historical_data.yml"
    CURRENT_BAR = ""
    BASE_SECONDS = 3
    INIT_REQUEST_ID = 1000
    SLEEP_DURATION = 1
    LOAD_DURATION = 10
    DATE_STR_POS = 8
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

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = Path(current_dir).parents[1]

        config_file_path = os.path.join(parent_dir, "config/", self.CONFIG_FILENAME)
        self.script_config = config_loader(config_file_path, validate=False)

        data_path = self.config["data_path"]
        saved_data_path = os.path.join(data_path, "saved_data/")
        pipeline_name = self.pipeline["pipeline"]["filename"]
        pipeline_data_path = os.path.join(saved_data_path, f"{pipeline_name}/")

        # Create log and data folders
        if not os.path.exists(saved_data_path):
            os.makedirs(saved_data_path)

        if not os.path.exists(pipeline_data_path):
            os.makedirs(pipeline_data_path)

        self.folder_name = pipeline_data_path

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

        self.bar_columns = self.script_config["bar_columns"]

        self.timezone = self.pipeline["pipeline"]["timezone"]

        self.historical_info = self.pipeline["pipeline"]["historical_data_config"]
        self.data_df = pd.DataFrame(columns=self.historical_info["columns"])
        self.data_list = []

        self.step_size = self.script_config["step_size"][
            self.historical_info["barSizeSetting"]
        ]

        self.max_loops = self.step_size["loops_required"]

        self.first_date = self.config["date_list"][0]
        self.date_list = self.config["date_list"][1:]

        self.timer = self.setTimer()

    def connect(self, ip_address: str, port: int, client_id: int) -> None:
        """Establish a broker connection for historical data retrieval.

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
                "Failed to connect for historical data collection",
                broker_name="interactive_brokers",
                host=ip_address,
                port=port,
                context={"client_id": client_id, "error": str(exc)},
            ) from exc

    def run(self) -> None:
        """Start the historical data collection workflow.

        Raises:
            BrokerConnectionError: If called before ``connect``.
        """
        if not self._connected:
            raise BrokerConnectionError(
                "PastData.run called before broker connection was established",
                broker_name="interactive_brokers",
                host=self.config.get("ip_address"),
                port=self.config.get("port"),
                context={
                    "client_id": self.pipeline["pipeline"]["client_id"]["historical"]
                },
            )
        self.start()

    def isConnected(self) -> bool:
        """Return whether the broker adapter is currently connected."""
        return self.adapter.is_connected()

    def disconnect(self) -> None:
        """Disconnect the broker adapter and reset connection state."""
        self.adapter.disconnect()
        self._connected = False

    def setTimer(self) -> int:
        """Calculate an estimated runtime in seconds for the collection."""
        num_dates = len(self.config["date_list"])
        loops_required = self.max_loops

        return self.BASE_SECONDS * num_dates * loops_required

    def checkDataframe(self, date_requested: str) -> bool:
        """Validate that the assembled dataframe matches the expected date.

        Args:
            date_requested: Date string in ``YYYYMMDD`` format.

        Returns:
            ``True`` if the dataframe has the expected row count and date.
        """
        if self.data_df.empty or self.DATE_COLUMN not in self.data_df.columns:
            return False

        dt_min = self.data_df[self.DATE_COLUMN].min()[: self.DATE_STR_POS]
        dt_max = self.data_df[self.DATE_COLUMN].max()[: self.DATE_STR_POS]
        row_check = (
            self.step_size["durationNum"] / self.step_size["barSize"]
        ) * self.step_size["loops_required"]

        if (
            dt_min == dt_max
            and self.data_df.shape[0] == row_check
            and dt_min == date_requested
        ):
            return True
        else:
            return False

    def adjustTime(self, time: datetime.datetime, minutes: int) -> datetime.datetime:
        """Subtract *minutes* from the given *time*."""
        return time - datetime.timedelta(minutes=minutes)

    def sendRequests(self, date: datetime.date) -> None:
        """Request historical bars for a single trading date.

        Loops through the configured request windows, collects bar
        data from the broker adapter, and finalizes the date's CSV
        output.

        Args:
            date: The trading date to collect data for.
        """
        self.end_date = datetime.datetime.combine(
            date,
            datetime.time(
                self.step_size["date_hour_max"],
                self.step_size["date_minute_max"],
                self.step_size["date_second_max"],
            ),
        )

        self.loop_it = 0
        self.data_list = []
        temp_time = self.end_date

        while self.loop_it < self.step_size["loops_required"]:
            adj_time = self.adjustTime(temp_time, self.step_size["increment_size"])

            try:
                bars = self.adapter.request_historical_data(
                    self.contract_spec,
                    temp_time,
                    self.step_size["durationString"],
                    self.historical_info["barSizeSetting"],
                    self.historical_info["whatToShow"],
                )
            except Exception as exc:
                raise BrokerConnectionError(
                    "Failed to request historical data from broker",
                    broker_name="interactive_brokers",
                    host=self.config.get("ip_address"),
                    port=self.config.get("port"),
                    context={
                        "request_id": self.req_it,
                        "symbol": self.contract_spec.symbol,
                        "bar_size": self.historical_info["barSizeSetting"],
                        "error": str(exc),
                    },
                ) from exc

            for bar in bars:
                self._consume_historical_bar(bar)

            self.loop_it += 1
            self.req_it += 1
            temp_time = adj_time
            time.sleep(self.SLEEP_DURATION)

        self._finalize_date_file(date)

    def _bar_to_row(self, bar: BarData) -> dict[str, object]:
        timestamp_text = bar.timestamp.strftime("%Y%m%d %H:%M:%S")
        return {
            self.bar_columns["bar_date"]: timestamp_text,
            self.bar_columns["bar_open"]: bar.open,
            self.bar_columns["bar_high"]: bar.high,
            self.bar_columns["bar_low"]: bar.low,
            self.bar_columns["bar_close"]: bar.close,
            self.bar_columns["bar_volume"]: bar.volume,
            self.bar_columns["bar_wap"]: bar.vwap,
            self.bar_columns["bar_barCount"]: bar.trade_count,
        }

    def _consume_historical_bar(self, bar: BarData) -> None:
        timestamp_text = bar.timestamp.strftime("%Y%m%d %H:%M:%S")

        if not self.CURRENT_BAR:
            self.CURRENT_BAR = timestamp_text
            return

        if self.CURRENT_BAR != timestamp_text:
            self.data_list.append(self._bar_to_row(bar))
            self.CURRENT_BAR = timestamp_text

    def _finalize_date_file(self, date: datetime.date) -> None:
        date_requested = date.strftime("%Y%m%d")
        self.CURRENT_BAR = ""

        self.data_df = pd.DataFrame(self.data_list)
        self.data_df = self.data_df.drop_duplicates(subset=self.DATE_COLUMN)
        self.data_df = self.data_df[self.historical_info["columns"]]
        self.data_df = self.data_df.fillna(0)
        self.data_df = self.data_df.sort_values(self.DATE_COLUMN)

        if self.checkDataframe(date_requested):
            self.data_df.to_csv(
                "{}/{}_{}_{}.csv".format(
                    self.folder_name,
                    self.contract_spec.symbol,
                    self.contract_spec.primary_exchange,
                    date_requested,
                ),
                index=False,
            )
            self.logger.info(
                "Historical data saved | symbol=%s date=%s bars=%s",
                self.contract_spec.symbol,
                date_requested,
                self.data_df.shape[0],
            )

        else:
            self.logger.warning(
                "Historical data validation failed | symbol=%s date=%s bars=%s",
                self.contract_spec.symbol,
                date_requested,
                self.data_df.shape[0],
            )
            raise DataError(
                "Date mismatch or incorrect row count in historical dataframe",
                data_source="historical_data",
                row_count=int(self.data_df.shape[0]),
                context={
                    "symbol": self.contract_spec.symbol,
                    "date_requested": date_requested,
                },
            )

        time.sleep(self.LOAD_DURATION)

        self.data_list = []

        if self.date_list:
            self.sendRequests(self.date_list.pop(0))
        else:
            self.stop()

    def start(self) -> None:
        """Begin the request loop for all configured dates."""
        self.req_it = self.INIT_REQUEST_ID

        # Request historical data for first date in the list
        self.sendRequests(self.first_date)

    def stop(self) -> None:
        """Mark collection as complete and disconnect the broker."""
        self.done = True
        self.adapter.disconnect()
        self._connected = False
