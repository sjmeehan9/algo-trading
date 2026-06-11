import type { AlgorithmDef, HyperparameterValue, SelectOption } from './algorithms';

export const DEFAULT_ML_ALGORITHM = 'random_forest';

export const ML_ALGORITHMS: AlgorithmDef[] = [
  {
    id: 'random_forest',
    name: 'Random Forest Classifier',
    description: 'Tree ensemble for tabular market, indicator, and sentiment features.',
    framework: 'sklearn',
    hyperparameters: [
      {
        name: 'n_estimators',
        label: 'Number of Trees',
        type: 'number',
        default: 100,
        min: 10,
        max: 1_000,
        step: 10,
        description: 'Number of decision trees in the ensemble.',
      },
      {
        name: 'max_depth',
        label: 'Max Depth',
        type: 'number',
        default: 10,
        min: 1,
        max: 50,
        step: 1,
        description: 'Maximum depth allowed for each tree.',
      },
    ],
  },
  {
    id: 'gradient_boosting',
    name: 'Gradient Boosting Classifier',
    description: 'Boosted tree ensemble for tabular market, indicator, and sentiment features.',
    framework: 'sklearn',
    hyperparameters: [
      {
        name: 'n_estimators',
        label: 'Number of Stages',
        type: 'number',
        default: 100,
        min: 10,
        max: 1_000,
        step: 10,
        description: 'Number of boosting stages to fit.',
      },
      {
        name: 'learning_rate',
        label: 'Learning Rate',
        type: 'number',
        default: 0.1,
        min: 0.001,
        max: 1,
        step: 0.001,
        description: 'Shrinkage applied to each boosting stage contribution.',
      },
      {
        name: 'max_depth',
        label: 'Max Depth',
        type: 'number',
        default: 3,
        min: 1,
        max: 20,
        step: 1,
        description: 'Maximum depth of each boosted tree.',
      },
    ],
  },
  {
    id: 'logistic_regression',
    name: 'Logistic Regression',
    description: 'Linear classifier baseline for tabular and sentiment features.',
    framework: 'sklearn',
    hyperparameters: [
      {
        name: 'C',
        label: 'Inverse Regularization (C)',
        type: 'number',
        default: 1,
        min: 0.001,
        max: 100,
        step: 0.001,
        description: 'Smaller values apply stronger regularization.',
      },
      {
        name: 'max_iter',
        label: 'Max Iterations',
        type: 'number',
        default: 1_000,
        min: 100,
        max: 10_000,
        step: 100,
        description: 'Maximum solver iterations before stopping.',
      },
    ],
  },
  {
    id: 'lstm',
    name: 'LSTM Neural Network',
    description: 'Sequence model for market and indicator time-series prediction.',
    framework: 'tensorflow',
    hyperparameters: [
      {
        name: 'units',
        label: 'LSTM Units',
        type: 'number',
        default: 64,
        min: 16,
        max: 512,
        step: 16,
        description: 'Hidden units in each recurrent layer.',
      },
      {
        name: 'num_layers',
        label: 'Number of Layers',
        type: 'number',
        default: 2,
        min: 1,
        max: 5,
        step: 1,
        description: 'Stacked recurrent layers used by the model.',
      },
      {
        name: 'dropout',
        label: 'Dropout Rate',
        type: 'number',
        default: 0.2,
        min: 0,
        max: 0.5,
        step: 0.05,
        description: 'Regularization applied between recurrent layers.',
      },
    ],
  },
  {
    id: 'mlp',
    name: 'MLP Neural Network',
    description: 'Feed-forward network for tabular market and indicator features.',
    framework: 'tensorflow',
    hyperparameters: [
      {
        name: 'units',
        label: 'Hidden Units',
        type: 'number',
        default: 64,
        min: 16,
        max: 512,
        step: 16,
        description: 'Units in each hidden dense layer.',
      },
      {
        name: 'num_layers',
        label: 'Number of Layers',
        type: 'number',
        default: 2,
        min: 1,
        max: 5,
        step: 1,
        description: 'Hidden dense layers used by the model.',
      },
      {
        name: 'dropout',
        label: 'Dropout Rate',
        type: 'number',
        default: 0.2,
        min: 0,
        max: 0.5,
        step: 0.05,
        description: 'Regularization applied between hidden layers.',
      },
    ],
  },
  {
    id: 'transformer_sentiment',
    name: 'Transformer Sentiment',
    description: 'Pre-trained language model for financial news sentiment signals.',
    framework: 'huggingface',
    hyperparameters: [
      {
        name: 'model_name',
        label: 'Pre-trained Model',
        type: 'select',
        default: 'finbert',
        options: [
          { value: 'finbert', label: 'FinBERT Financial' },
          { value: 'roberta-sentiment', label: 'RoBERTa Sentiment' },
          { value: 'distilbert', label: 'DistilBERT Fast' },
        ],
        description: 'Model checkpoint used for text embeddings and classification.',
      },
      {
        name: 'max_length',
        label: 'Max Sequence Length',
        type: 'number',
        default: 512,
        min: 64,
        max: 1_024,
        step: 64,
        description: 'Maximum token length accepted for each news item.',
      },
    ],
  },
  {
    id: 'finbert',
    name: 'FinBERT Sentiment',
    description: 'Pretrained FinBERT checkpoint for financial news sentiment signals.',
    framework: 'huggingface',
    hyperparameters: [
      {
        name: 'max_length',
        label: 'Max Sequence Length',
        type: 'number',
        default: 512,
        min: 64,
        max: 1_024,
        step: 64,
        description: 'Maximum token length accepted for each news item.',
      },
    ],
  },
  {
    id: 'vader',
    name: 'VADER Sentiment',
    description: 'Rule-based lexicon sentiment scorer; no training required.',
    framework: 'huggingface',
    hyperparameters: [],
  },
];

/** Return a supported ML algorithm definition, falling back to random forest. */
export const getMLAlgorithmDefinition = (algorithmId: string): AlgorithmDef =>
  ML_ALGORITHMS.find((algorithm) => algorithm.id === algorithmId) ?? ML_ALGORITHMS[0]!;

/** Build default hyperparameter values for the selected ML algorithm. */
export const getDefaultMLHyperparameters = (
  algorithmId: string = DEFAULT_ML_ALGORITHM,
): Record<string, HyperparameterValue> =>
  Object.fromEntries(
    getMLAlgorithmDefinition(algorithmId).hyperparameters.map((definition) => [
      definition.name,
      definition.default,
    ]),
  );

/** Return framework select options represented by a list of algorithm definitions. */
export const getFrameworkOptions = (algorithms: AlgorithmDef[]): SelectOption[] => {
  const frameworks = [...new Set(algorithms.map((algorithm) => algorithm.framework))];
  return frameworks.map((framework) => ({
    value: framework,
    label: framework,
  }));
};

/** Return algorithms belonging to a specific trainer framework. */
export const getAlgorithmsForFramework = (
  algorithms: AlgorithmDef[],
  framework: string,
): AlgorithmDef[] => algorithms.filter((algorithm) => algorithm.framework === framework);
