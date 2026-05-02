import datetime
import logging
from threading import Timer

import pandas as pd
from algotrading.src.broker import (
    AccountInfo,
    BarData,
    BrokerAdapter,
    ContractSpec,
    InstrumentType,
    InteractiveBrokersAdapter,
    OrderStatus,
    PositionInfo,
)
from algotrading.src.trading.deployment import validate_pipeline_deployment_config
from algotrading.src.trading.inference import RealTimeInferencePipeline, TradingDecision

from ..exceptions import BrokerConnectionError
from ..models.predict import Predict
from .order import OrderManager
from .payload import Payload
from .tools import TradingTools

logger = logging.getLogger(__name__)


class Trading:
    """Orchestrates live trading through a broker adapter.

    Routes order placement, cancellation, and account/position updates
    through an injected ``BrokerAdapter``, preserving the existing position
    state-machine and prediction-driven trading algorithm.
    """

    ELIGABLE_STREAM = "real"
    TRADE_TIMER = 4
    CURRENT_POS_LIST = []
    ACTIONS = {0: "NONE", 1: "BUY", 2: "SELL", "NONE": 0, "BUY": 1, "SELL": 2}

    def __init__(
        self,
        config: dict,
        pipeline: dict,
        adapter: BrokerAdapter | None = None,
        inference_pipeline: RealTimeInferencePipeline | None = None,
    ):
        """Initialise trading session.

        Args:
            config: Application configuration dictionary.
            pipeline: Pipeline configuration dictionary.
            adapter: Broker adapter instance; defaults to
                ``InteractiveBrokersAdapter`` when *None*.
            inference_pipeline: Optional real-time inference pipeline used for
                core RL decisions with supporting model signals.
        """
        self.logger = logger

        self.config = config
        self.pipeline = pipeline
        self.adapter = adapter if adapter is not None else InteractiveBrokersAdapter()
        self.inference_pipeline = inference_pipeline

        self.predict = (
            Predict(self.config, self.pipeline)
            if self.inference_pipeline is None
            else None
        )
        self.order = OrderManager(self.config, self.pipeline)
        self.tools = TradingTools(self.pipeline)
        self.payload = Payload()
        self.payload.action_dict = self.ACTIONS
        self._connected = False
        self._callbacks_registered = False
        self.timing: Timer | None = None
        self.timer = False

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
                float(contract_info["strike"])
                if contract_info.get("strike") is not None
                else None
            ),
            right=contract_info.get("right"),
            multiplier=(
                float(contract_info["multiplier"])
                if contract_info.get("multiplier") is not None
                else 1.0
            ),
        )

        self.account = self.config["account_number"]
        self.enable_trading = self.config["stream_data"] == self.ELIGABLE_STREAM
        if self.enable_trading:
            validate_pipeline_deployment_config(
                self.pipeline,
                deployment_mode=str(self.config.get("deployment_mode", "paper")),
            )
        self.logger.info(f"Enable trading: {self.enable_trading}")

        client_id = self.pipeline["pipeline"]["client_id"]
        try:
            self.adapter.connect(
                self.config["ip_address"],
                self.config["port"],
                client_id["trading"],
            )
            self._connected = True
        except Exception as exc:
            raise BrokerConnectionError(
                "Failed to connect trading client",
                broker_name="interactive_brokers",
                host=self.config.get("ip_address"),
                port=self.config.get("port"),
                context={"client_id": client_id.get("trading"), "error": str(exc)},
            ) from exc

        self.start()

        self.runtime = self.setTimer()
        Timer(self.runtime, self.stop).start()

    def setTimer(self) -> int:
        """Return the configured runtime duration in seconds."""

        runtime = self.pipeline["pipeline"]["live_data_config"]["runtime"]
        return runtime

    def _on_account_update(self, account_info: AccountInfo) -> None:
        """Handle account snapshot updates from the broker adapter."""

        if account_info.currency == self.contract_spec.currency:
            self.payload.cashbalance = float(account_info.cash_balance)

            self.logger.info(f"cashbalance: {self.payload.cashbalance}")

            if "_FILL" in self.payload.active_pos:
                self.payload.update_state_data, self.payload.current_pos_list = (
                    self.order.positionUnlock(
                        self.payload.active_pos,
                        self.payload.cashbalance,
                        self.payload.current_pos_list,
                    )
                )

    def _on_position_update(self, position: PositionInfo) -> None:
        """Handle position updates from the broker adapter."""

        if position.contract.symbol == self.contract_spec.symbol:
            self.payload.openunits = float(position.quantity)

            self.logger.info(f"openunits: {self.payload.openunits}")

            if "_FILL" in self.payload.active_pos:
                self.payload.update_state_data, self.payload.current_pos_list = (
                    self.order.positionUnlock(
                        self.payload.active_pos,
                        self.payload.openunits,
                        self.payload.current_pos_list,
                    )
                )

    def confirmTrades(self) -> None:
        """Release trade lock when state data has been updated."""

        if self.payload.update_state_data:
            self.payload.release_trade = True
            self.payload.update_state_data = False

        return None

    def _on_order_status(self, status_update: OrderStatus) -> None:
        """Handle order status transitions from the broker adapter."""

        if len(self.payload.order_spec) < 3:
            return

        status = status_update.status
        filled = status_update.filled_quantity
        avg_fill_price = status_update.average_fill_price

        self.logger.info(
            "OrderStatus. Id: %s, Status: %s, %s, %s, %s",
            status_update.order_id,
            status,
            status_update.filled_quantity,
            status_update.remaining_quantity,
            status_update.average_fill_price,
        )

        if (
            (status == "PENDING" or status == "SUBMITTED")
            and filled == 0
            and self.payload.order_spec[2] == "open"
        ):
            self.timing = Timer(self.TRADE_TIMER, self.stopCancel, args=[self.oid])
            self.timing.start()
            self.timer = True

            self.logger.info(
                f"PEND, {self.payload.active_pos}, {self.payload.last_pos}, {self.payload.last_price}"
            )

        elif (
            (status == "SUBMITTED" or status == "PARTIAL")
            and filled > 0
            and self.payload.order_spec[2] == "open"
        ):
            self.payload.last_price = float(avg_fill_price or self.payload.last_price)
            self.payload.last_pos = self.payload.order_spec[0]
            self.payload.active_pos = "{}_PART".format(self.payload.temp_action)

            self.logger.info(
                f"PART, {self.payload.active_pos}, {self.payload.last_pos}, {self.payload.last_price}"
            )

        elif status == "FILLED":
            self.payload.active_pos = "{}_FILL".format(self.payload.temp_action)
            if self.timing is not None:
                self.timing.cancel()
            self.timer = False

            self.payload.last_price = float(avg_fill_price or self.payload.last_price)
            self.payload.last_pos = self.payload.order_spec[0]

            self.logger.info(
                f"FILL, {self.payload.active_pos}, {self.payload.last_pos}, {self.payload.last_price}"
            )

    def stopCancel(self, orderId: str) -> None:
        """Cancel a pending or partially-filled order via the adapter."""

        self.logger.info(f"order cancelled: {orderId}, {self.payload.active_pos}")

        if "_PEND" in self.payload.active_pos or "_PART" in self.payload.active_pos:
            self.adapter.cancel_order(str(orderId))
            self.timer = False

            if "_PEND" in self.payload.active_pos:
                self.payload.active_pos = self.payload.last_pos

            if "_PART" in self.payload.active_pos:
                self.payload.active_pos = "{}_FILL".format(self.payload.temp_action)

    def tradingAlgorithm(self, state: dict, state_df: pd.DataFrame) -> None:
        """Run the prediction-driven trading algorithm for a single tick."""

        self.logger.info(f"Pre action payload: {self.payload}")

        if self.payload.release_trade == True:
            self.payload.active_pos = self.payload.last_pos
            self.payload.release_trade = False

        if self.inference_pipeline is None:
            if self.predict is None:
                raise RuntimeError("Predictor is not configured")
            action, _ = self.predict.get_action(state)
            self.payload.action_int = action.item()
            prediction_log_value = action.item()
        else:
            decision = self._run_realtime_inference(state_df)
            min_confidence = self._get_min_inference_confidence()
            action_str = (
                decision.action
                if decision.should_execute(min_confidence=min_confidence)
                else "HOLD"
            )
            self.payload.action_int = self._decision_action_to_int(action_str)
            self.payload.last_decision = decision
            prediction_log_value = decision.to_dict()

        self.payload.action_int = self.tools.stop_take(self.payload.action_int, state)

        self.payload.action_str = self.payload.action_dict[self.payload.action_int]

        self.logger.info(
            f"action taken: {self.payload.action_str}, prediction: {prediction_log_value}"
        )

        take_action = self.order.checkAction(
            self.payload.action_str, self.payload.active_pos
        )

        self.logger.info("End of trade data flow: %s", datetime.datetime.now())

        if take_action and self.enable_trading:
            self.logger.info("Action requested")
            self.payload.live_price = self.order.priceAction(state_df)
            self.executeOrder()
        else:
            self.logger.info("No action taken")
            self.payload.action_int = 0
            self.payload.action_str = self.payload.action_dict[self.payload.action_int]

        return None

    def _run_realtime_inference(self, state_df: pd.DataFrame) -> TradingDecision:
        """Run the injected real-time pipeline against the latest market row."""

        if self.inference_pipeline is None:
            raise RuntimeError("inference_pipeline is not configured")
        if not self.inference_pipeline.is_running():
            self.inference_pipeline.start_sync()
        return self.inference_pipeline.process_market_data_sync(
            self._state_df_to_bar_data(state_df)
        )

    def _state_df_to_bar_data(self, state_df: pd.DataFrame) -> BarData:
        """Convert the latest live state dataframe row to broker-normalized bar data."""

        if state_df.empty:
            raise ValueError("state_df must include at least one row")

        row = state_df.iloc[-1]
        timestamp = self._coerce_bar_timestamp(row["date"])
        return BarData(
            timestamp=timestamp,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
            vwap=float(row["wap"] if "wap" in row else row.get("vwap", 0.0)),
            trade_count=int(
                row["count"] if "count" in row else row.get("trade_count", 0)
            ),
        )

    def _coerce_bar_timestamp(self, value: object) -> datetime.datetime:
        """Convert dataframe timestamp values to timezone-aware datetimes."""

        if isinstance(value, pd.Timestamp):
            timestamp = value.to_pydatetime()
        elif isinstance(value, datetime.datetime):
            timestamp = value
        elif isinstance(value, (int, float)):
            timestamp = datetime.datetime.fromtimestamp(
                float(value),
                tz=datetime.timezone.utc,
            )
        else:
            timestamp = pd.Timestamp(value).to_pydatetime()

        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            return timestamp.replace(tzinfo=datetime.timezone.utc)
        return timestamp

    def _decision_action_to_int(self, action: str) -> int:
        """Convert a real-time decision action string to legacy action integer."""

        if action == "HOLD":
            return self.ACTIONS["NONE"]
        return int(self.ACTIONS.get(action, self.ACTIONS["NONE"]))

    def _get_min_inference_confidence(self) -> float:
        """Return configured confidence threshold for real-time decisions."""

        trading_config = self.pipeline.get("pipeline", {}).get("trading_config", {})
        raw_value = trading_config.get("min_confidence", 0.5)
        return float(raw_value)

    def executeOrder(self) -> None:
        """Build and submit an order through the broker adapter."""

        self.payload.previous_pos = self.payload.active_pos

        self.payload.temp_action = self.payload.action_str

        self.payload.temp_action_int = self.payload.action_int

        self.payload.order_spec = self.order.calcOrderSpec(
            self.payload.cashbalance,
            self.payload.openunits,
            self.payload.temp_action,
            self.payload.active_pos,
            self.payload.live_price,
        )

        order, self.payload.active_pos = self.order.buildOrder(
            self.payload.temp_action, self.payload.order_spec[1]
        )

        try:
            self.oid = self.adapter.place_order(self.contract_spec, order)
        except Exception as exc:
            self.logger.error("Order placement failed")
            raise BrokerConnectionError(
                "Order placement failed",
                broker_name="interactive_brokers",
                host=self.config.get("ip_address"),
                port=self.config.get("port"),
                context={
                    "symbol": self.contract_spec.symbol,
                    "action": self.payload.temp_action,
                    "order_spec": self.payload.order_spec,
                    "error": str(exc),
                },
            ) from exc

        return None

    def start(self) -> None:
        """Register adapter callbacks and hydrate initial account state."""

        self.logger.info("Starting trading callbacks")

        if not self._callbacks_registered:
            self.adapter.register_order_callback(self._on_order_status)
            self.adapter.subscribe_account_updates(self._on_account_update)
            self.adapter.subscribe_position_updates(self._on_position_update)
            self._callbacks_registered = True

        try:
            account = self.adapter.get_account_info()
            self._on_account_update(account)
        except Exception:
            self.logger.info("Account snapshot not yet available; waiting for callback")

        for position in self.adapter.get_positions():
            self._on_position_update(position)

    def stop(self) -> None:
        """Disconnect from the broker adapter."""

        self.logger.info("Trading connection closed")

        if self.inference_pipeline is not None and self.inference_pipeline.is_running():
            self.inference_pipeline.stop_sync()

        if self._connected:
            self.adapter.disconnect()
            self._connected = False
