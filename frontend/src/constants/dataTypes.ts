import type { SelectOption } from './algorithms';

export type InputDataTypeIcon = 'market' | 'news' | 'indicator';

export interface InputDataTypeDef {
  id: string;
  label: string;
  description: string;
  icon: InputDataTypeIcon;
  frequencies: string[];
}

export interface SignalTypeDef {
  id: string;
  label: string;
  description: string;
  valueRange: string;
}

export const DEFAULT_SUPPORTING_INPUT_FREQUENCY = '1m';

export const DEFAULT_SUPPORTING_SIGNAL_TYPE = 'sentiment';

export const FORBIDDEN_SUPPORTING_INPUT_TYPES = [
  'model_signal',
  'supporting_model_signal',
  'signal',
];

export const INPUT_DATA_TYPES: InputDataTypeDef[] = [
  {
    id: 'market_bar',
    label: 'Market Data (OHLCV)',
    description: 'Price and volume bars from broker or file sources.',
    icon: 'market',
    frequencies: ['5s', '1m', '5m', '15m', '1h', '1d'],
  },
  {
    id: 'news_text',
    label: 'News Text',
    description: 'Headlines and article text from configured news providers.',
    icon: 'news',
    frequencies: ['realtime', 'hourly', 'daily'],
  },
  {
    id: 'technical_indicator',
    label: 'Technical Indicators',
    description: 'Precomputed indicator features aligned with market data.',
    icon: 'indicator',
    frequencies: ['same as market data'],
  },
];

export const SUPPORTING_INPUT_FREQUENCIES: SelectOption[] = [
  { value: '5s', label: '5 seconds' },
  { value: '1m', label: '1 minute' },
  { value: '5m', label: '5 minutes' },
  { value: '15m', label: '15 minutes' },
  { value: '1h', label: '1 hour' },
  { value: '1d', label: '1 day' },
  { value: 'realtime', label: 'Real-time' },
  { value: 'same as market data', label: 'Same as market data' },
];

export const SIGNAL_TYPES: SignalTypeDef[] = [
  {
    id: 'sentiment',
    label: 'Sentiment',
    description: 'Directional tone from bearish to bullish.',
    valueRange: '[-1, 1]',
  },
  {
    id: 'trend',
    label: 'Trend',
    description: 'Trend direction and strength.',
    valueRange: '[-1, 1]',
  },
  {
    id: 'volatility',
    label: 'Volatility',
    description: 'Predicted volatility level.',
    valueRange: '[0, infinity)',
  },
  {
    id: 'position',
    label: 'Position',
    description: 'Position recommendation for the core model.',
    valueRange: '{0, 1, 2}',
  },
  {
    id: 'indicator',
    label: 'Custom Indicator',
    description: 'Numeric indicator value from a custom model.',
    valueRange: 'varies',
  },
  {
    id: 'custom',
    label: 'Custom Signal',
    description: 'Domain-specific signal consumed by a compatible core model.',
    valueRange: 'configured',
  },
];

/** Return true when the input data type can be selected for supporting models. */
export const isSelectableSupportingInputType = (dataTypeId: string): boolean =>
  INPUT_DATA_TYPES.some((dataType) => dataType.id === dataTypeId.trim().toLowerCase()) &&
  !FORBIDDEN_SUPPORTING_INPUT_TYPES.includes(dataTypeId.trim().toLowerCase());

/** Return true when the signal type is exposed by the supporting-model UI. */
export const isSupportedSignalType = (signalTypeId: string): boolean =>
  SIGNAL_TYPES.some((signalType) => signalType.id === signalTypeId);
