"""
=========================================================
Utility Functions
California Housing Price Predictor
=========================================================

This module contains reusable helper functions used across
the project.

Functions Included
------------------
• Price formatting
• Price category classification
• Luxury score calculation
• Premium house detection
• Prediction interval calculation
• Timestamp generation
• Console gauges
"""

from __future__ import annotations

from datetime import datetime
import numpy as np


class Utils:
    """
    Collection of reusable static utility methods.
    """

    # ---------------------------------------------------------
    # Formatting
    # ---------------------------------------------------------

    @staticmethod
    def format_currency(value: float) -> str:
        """
        Format number as US currency.

        Example
        -------
        456832.123

        becomes

        $456,832.12
        """

        return f"${value:,.2f}"

    @staticmethod
    def current_timestamp() -> str:
        """
        Return current timestamp.
        """

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---------------------------------------------------------
    # Price Category
    # ---------------------------------------------------------

    @staticmethod
    def price_category(price: float) -> str:
        """
        Categorize predicted house price.
        """

        if price < 150000:
            return "Budget"

        elif price < 300000:
            return "Mid Range"

        elif price < 450000:
            return "Premium"

        else:
            return "Luxury"

    # ---------------------------------------------------------
    # Premium House
    # ---------------------------------------------------------

    @staticmethod
    def is_premium(price: float) -> bool:
        """
        Whether the predicted house belongs
        to premium price segment.
        """

        return price >= 300000

    # ---------------------------------------------------------
    # Luxury Score
    # ---------------------------------------------------------

    @staticmethod
    def luxury_score(
        predicted_price: float,
        median_income: float,
        rooms_per_household: float,
        bedrooms_per_room: float,
        ocean_proximity: str,
    ) -> int:
        """
        Estimate a luxury score (0-100).

        This is NOT the model prediction.
        It is simply a quality indicator.
        """

        score = 0

        # Price Contribution (40)

        score += min(predicted_price / 500000 * 40, 40)

        # Income Contribution (25)

        score += min(median_income / 10 * 25, 25)

        # Spacious Houses (15)

        score += min(rooms_per_household / 10 * 15, 15)

        # Bedroom Ratio (10)

        if bedrooms_per_room < 0.30:
            score += 10

        elif bedrooms_per_room < 0.40:
            score += 7

        elif bedrooms_per_room < 0.50:
            score += 4

        # Ocean Bonus (10)

        premium_locations = {
            "NEAR BAY",
            "NEAR OCEAN",
            "<1H OCEAN",
        }

        if ocean_proximity in premium_locations:
            score += 10

        return int(min(score, 100))

    # ---------------------------------------------------------
    # Prediction Interval
    # ---------------------------------------------------------

    @staticmethod
    def prediction_interval(model, X):
        """
        Estimate prediction uncertainty
        for Random Forest models.

        Returns
        -------
        mean_prediction
        lower_bound
        upper_bound
        standard_deviation
        """

        if not hasattr(model, "estimators_"):
            prediction = model.predict(X)[0]

            return (
                prediction,
                prediction,
                prediction,
                0.0,
            )

        tree_predictions = np.array([
            tree.predict(X)[0]
            for tree in model.estimators_
        ])

        mean_prediction = tree_predictions.mean()

        std = tree_predictions.std()

        lower = mean_prediction - 1.96 * std

        upper = mean_prediction + 1.96 * std

        return (
            mean_prediction,
            lower,
            upper,
            std,
        )

    # ---------------------------------------------------------
    # Console Gauge
    # ---------------------------------------------------------

    @staticmethod
    def price_gauge(price: float) -> str:
        """
        Create text progress bar.

        Example

        ████████████--------
        """

        maximum_price = 600000

        percentage = min(price / maximum_price, 1.0)

        blocks = int(percentage * 20)

        return "█" * blocks + "-" * (20 - blocks)

    # ---------------------------------------------------------
    # Color Selection
    # ---------------------------------------------------------

    @staticmethod
    def marker_color(price: float) -> str:
        """
        Color for Folium marker.
        """

        if price >= 450000:
            return "red"

        elif price >= 300000:
            return "orange"

        else:
            return "green"

    # ---------------------------------------------------------
    # Report Dictionary
    # ---------------------------------------------------------

    @staticmethod
    def build_summary(
        prediction: float,
        category: str,
        luxury_score: int,
        premium: bool,
        interval: tuple,
    ) -> dict:
        """
        Build standardized prediction summary.
        """

        mean, lower, upper, std = interval

        return {
            "prediction": prediction,
            "prediction_formatted": Utils.format_currency(prediction),
            "category": category,
            "luxury_score": luxury_score,
            "premium_house": premium,
            "prediction_interval": (
                Utils.format_currency(lower),
                Utils.format_currency(upper),
            ),
            "standard_deviation": round(std, 2),
            "timestamp": Utils.current_timestamp(),
        }