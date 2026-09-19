"""Inventory AI public API."""

from inventory_ai.forecasting import forecast_demand
from inventory_ai.normalization import normalize_movement
from inventory_ai.qa import answer_question

__all__ = ["answer_question", "forecast_demand", "normalize_movement"]
