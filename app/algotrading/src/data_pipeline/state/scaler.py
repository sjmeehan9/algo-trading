"""Scaler abstraction for data pipeline state management."""

from __future__ import annotations

import logging

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class ScalerWrapper:
    """Factory-backed scaler wrapper for state feature normalization.

    Args:
        scaler_type: Name of the scaler implementation to use.
    """

    _SUPPORTED_SCALERS = {
        "MinMaxScaler": MinMaxScaler,
        "StandardScaler": StandardScaler,
        "RobustScaler": RobustScaler,
    }

    def __init__(self, scaler_type: str) -> None:
        self.logger = logging.getLogger(__name__)
        self.scaler_type = scaler_type

        scaler_class = self._SUPPORTED_SCALERS.get(scaler_type)
        if scaler_class is None:
            raise ValueError(
                f"Unsupported scaler_type '{scaler_type}'. "
                f"Supported values: {sorted(self._SUPPORTED_SCALERS)}"
            )
        self._scaler = scaler_class()

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit scaler on data and return transformed values."""

        return self._scaler.fit_transform(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using already-fitted scaler."""

        return self._scaler.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled values back to original space."""

        return self._scaler.inverse_transform(data)
