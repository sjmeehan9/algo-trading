"""Factories for trainer implementations used by model orchestration."""

from __future__ import annotations

from algotrading.src.trainers import RLTrainer, SB3Algorithm, StableBaselines3Trainer


def create_rl_trainer(pipeline: dict) -> RLTrainer:
    """Create an RL trainer from pipeline model configuration.

    Args:
        pipeline: Pipeline configuration dictionary.

    Returns:
        A concrete RL trainer implementation.

    Raises:
        ValueError: If the configured model type is unsupported.
    """

    model_type = str(pipeline["pipeline"]["model"]["model_type"]).lower()
    model_policy = pipeline["pipeline"]["model"]["model_policy"]

    if model_type == "ppo":
        algorithm = SB3Algorithm.PPO
    elif model_type == "dqn":
        algorithm = SB3Algorithm.DQN
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return StableBaselines3Trainer(algorithm=algorithm, policy=model_policy)
