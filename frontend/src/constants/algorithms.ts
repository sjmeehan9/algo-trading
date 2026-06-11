export type HyperparameterValue = number | string | boolean;

export type HyperparameterInputType = 'number' | 'select' | 'boolean';

export interface HyperparameterOption {
  value: string;
  label: string;
}

export interface HyperparameterDef {
  name: string;
  label: string;
  type: HyperparameterInputType;
  default: HyperparameterValue;
  min?: number;
  max?: number;
  step?: number;
  options?: HyperparameterOption[];
  description: string;
}

export interface AlgorithmDef {
  id: string;
  name: string;
  description: string;
  framework: string;
  hyperparameters: HyperparameterDef[];
}

export interface RewardFunctionDef {
  id: string;
  name: string;
  description: string;
}

export interface SelectOption {
  value: string;
  label: string;
}

export interface EnvironmentDefaults {
  action_space: string;
  observation_window: number;
  initial_cash: number;
  transaction_cost_bps: number;
  max_position_size: number;
  allow_short_selling: boolean;
}

export const DEFAULT_RL_ALGORITHM = 'ppo';

export const DEFAULT_TRAINING_SYMBOLS = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA'];

export const DATA_FREQUENCIES: SelectOption[] = [
  { value: '5s', label: '5 seconds' },
  { value: '1m', label: '1 minute' },
  { value: '5m', label: '5 minutes' },
  { value: '15m', label: '15 minutes' },
  { value: '1h', label: '1 hour' },
  { value: '1d', label: '1 day' },
];

export const ACTION_SPACE_OPTIONS: SelectOption[] = [
  { value: 'discrete', label: 'Discrete buy/hold/sell' },
  { value: 'continuous', label: 'Continuous allocation' },
];

/** Default exchange-local session window (US equities regular trading hours). */
export const DEFAULT_SESSION_START = '09:30';
export const DEFAULT_SESSION_END = '15:30';
export const DEFAULT_SESSION_TIMEZONE = 'America/New_York';

/** Common exchange timezones offered for the trading-session window. */
export const SESSION_TIMEZONES: SelectOption[] = [
  { value: 'America/New_York', label: 'New York (US Eastern)' },
  { value: 'America/Chicago', label: 'Chicago (US Central)' },
  { value: 'America/Los_Angeles', label: 'Los Angeles (US Pacific)' },
  { value: 'Europe/London', label: 'London' },
  { value: 'Europe/Frankfurt', label: 'Frankfurt' },
  { value: 'Asia/Tokyo', label: 'Tokyo' },
  { value: 'Asia/Hong_Kong', label: 'Hong Kong' },
  { value: 'Australia/Sydney', label: 'Sydney' },
  { value: 'UTC', label: 'UTC' },
];

export const REWARD_FUNCTIONS: RewardFunctionDef[] = [
  {
    id: 'profit_seeker',
    name: 'Profit Seeker',
    description: 'Optimizes cumulative profit after spreads and transaction costs.',
  },
  {
    id: 'risk_adjusted',
    name: 'Risk Adjusted',
    description: 'Balances return against volatility and drawdown pressure.',
  },
  {
    id: 'sharpe_reward',
    name: 'Sharpe Ratio',
    description: 'Rewards stable returns relative to realized volatility.',
  },
];

export const DEFAULT_ENVIRONMENT_CONFIG: EnvironmentDefaults = {
  action_space: 'discrete',
  observation_window: 60,
  initial_cash: 100_000,
  transaction_cost_bps: 1,
  max_position_size: 1,
  allow_short_selling: false,
};

export const RL_ALGORITHMS: AlgorithmDef[] = [
  {
    id: 'ppo',
    name: 'PPO (Proximal Policy Optimization)',
    description: 'Stable on-policy training that is a strong default for trading environments.',
    framework: 'stable_baselines3',
    hyperparameters: [
      {
        name: 'learning_rate',
        label: 'Learning Rate',
        type: 'number',
        default: 0.0003,
        min: 0.000001,
        max: 0.1,
        step: 0.0001,
        description: 'Step size for gradient updates.',
      },
      {
        name: 'n_steps',
        label: 'Steps per Update',
        type: 'number',
        default: 2048,
        min: 64,
        max: 8192,
        step: 64,
        description: 'Environment steps collected before each policy update.',
      },
      {
        name: 'batch_size',
        label: 'Batch Size',
        type: 'number',
        default: 64,
        min: 8,
        max: 512,
        step: 8,
        description: 'Minibatch size used during policy optimization.',
      },
      {
        name: 'n_epochs',
        label: 'Epochs per Update',
        type: 'number',
        default: 10,
        min: 1,
        max: 50,
        step: 1,
        description: 'Optimization passes over each collected rollout.',
      },
      {
        name: 'gamma',
        label: 'Discount Factor',
        type: 'number',
        default: 0.99,
        min: 0.9,
        max: 0.9999,
        step: 0.001,
        description: 'Future reward discount applied during return estimation.',
      },
      {
        name: 'clip_range',
        label: 'Clip Range',
        type: 'number',
        default: 0.2,
        min: 0.1,
        max: 0.4,
        step: 0.05,
        description: 'Policy update clipping threshold for PPO stability.',
      },
    ],
  },
  {
    id: 'dqn',
    name: 'DQN (Deep Q-Network)',
    description: 'Off-policy value learning with replay memory for discrete actions.',
    framework: 'stable_baselines3',
    hyperparameters: [
      {
        name: 'learning_rate',
        label: 'Learning Rate',
        type: 'number',
        default: 0.0001,
        min: 0.000001,
        max: 0.1,
        step: 0.0001,
        description: 'Step size for Q-network updates.',
      },
      {
        name: 'buffer_size',
        label: 'Replay Buffer Size',
        type: 'number',
        default: 1_000_000,
        min: 10_000,
        max: 10_000_000,
        step: 10_000,
        description: 'Maximum stored transitions for replay sampling.',
      },
      {
        name: 'batch_size',
        label: 'Batch Size',
        type: 'number',
        default: 32,
        min: 8,
        max: 256,
        step: 8,
        description: 'Number of replay samples per training update.',
      },
      {
        name: 'gamma',
        label: 'Discount Factor',
        type: 'number',
        default: 0.99,
        min: 0.9,
        max: 0.9999,
        step: 0.001,
        description: 'Future reward discount applied to Q targets.',
      },
      {
        name: 'exploration_fraction',
        label: 'Exploration Fraction',
        type: 'number',
        default: 0.1,
        min: 0.01,
        max: 0.5,
        step: 0.01,
        description: 'Fraction of training used to decay random exploration.',
      },
      {
        name: 'target_update_interval',
        label: 'Target Update Interval',
        type: 'number',
        default: 10_000,
        min: 100,
        max: 100_000,
        step: 100,
        description: 'Steps between target network synchronization.',
      },
    ],
  },
  {
    id: 'a2c',
    name: 'A2C (Advantage Actor-Critic)',
    description: 'Synchronous actor-critic training with compact update batches.',
    framework: 'stable_baselines3',
    hyperparameters: [
      {
        name: 'learning_rate',
        label: 'Learning Rate',
        type: 'number',
        default: 0.0007,
        min: 0.000001,
        max: 0.1,
        step: 0.0001,
        description: 'Step size for actor and critic updates.',
      },
      {
        name: 'n_steps',
        label: 'Steps per Update',
        type: 'number',
        default: 5,
        min: 1,
        max: 2048,
        step: 1,
        description: 'Environment steps collected before each update.',
      },
      {
        name: 'gamma',
        label: 'Discount Factor',
        type: 'number',
        default: 0.99,
        min: 0.9,
        max: 0.9999,
        step: 0.001,
        description: 'Future reward discount used for advantage estimates.',
      },
      {
        name: 'ent_coef',
        label: 'Entropy Coefficient',
        type: 'number',
        default: 0,
        min: 0,
        max: 0.1,
        step: 0.001,
        description: 'Exploration bonus weight in the policy objective.',
      },
    ],
  },
];

/** Return the configured RL algorithm definition, falling back to PPO. */
export const getAlgorithmDefinition = (algorithmId: string): AlgorithmDef =>
  RL_ALGORITHMS.find((algorithm) => algorithm.id === algorithmId) ?? RL_ALGORITHMS[0]!;

/** Build default hyperparameter values for the selected RL algorithm. */
export const getDefaultHyperparameters = (
  algorithmId: string = DEFAULT_RL_ALGORITHM,
): Record<string, HyperparameterValue> =>
  Object.fromEntries(
    getAlgorithmDefinition(algorithmId).hyperparameters.map((definition) => [
      definition.name,
      definition.default,
    ]),
  );
