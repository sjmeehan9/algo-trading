from dataclasses import dataclass, field

@dataclass
class Payload:
    cashbalance: float = field(default=0.0)
    openunits: float = field(default=0.0)
    active_pos: str = field(default='NONE')
    last_price: float = field(default=0.0)
    live_price: float = field(default=0.0)
    last_pos: str = field(default='NONE')
    previous_pos: str = field(default='NONE')
    order_spec: list = field(default_factory=list)
    current_pos_list: list = field(default_factory=list)
    update_reward_vars: bool = field(default=False)
    release_trade: bool = field(default=False)
    action_int: int = field(default=0)
    action_str: str = field(default='NONE')
    temp_action: str = field(default='NONE')
    temp_action_int: int = field(default=0)
    action_dict: dict = field(default_factory=dict)