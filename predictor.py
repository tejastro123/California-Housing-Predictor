"""
=========================================================
Prediction Engine
California Housing Price Predictor
=========================================================

This module contains the core prediction engine.

Responsibilities
----------------
✓ Load trained model
✓ Validate user input
✓ Engineer features
✓ Perform prediction
✓ Estimate prediction interval
✓ Compute analytics
✓ Generate prediction summary
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import joblib

from validation import InputValidator
from feature_engineering import FeatureEngineer
from utils import Utils


class HousingPricePredictor:
    """
    Production-ready inference engine.
    """

    # --------------------------------------------------

    def __init__(self, model_path):

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found:\n{self.model_path}"
            )

        self.model = joblib.load(self.model_path)

        self.feature_engineer = FeatureEngineer(self.model)

    # --------------------------------------------------

    @property
    def model_name(self):

        return self.model.__class__.__name__

    # --------------------------------------------------

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete prediction pipeline.

        Returns
        -------
        Dictionary containing all prediction information.
        """

        # -------------------------------
        # Validation
        # -------------------------------

        cleaned_data = InputValidator.validate_and_clean(
            input_data
        )

        # -------------------------------
        # Feature Engineering
        # -------------------------------

        features = self.feature_engineer.transform(
            cleaned_data
        )

        analytics = self.feature_engineer.analytics(
            features
        )

        # -------------------------------
        # Prediction
        # -------------------------------

        prediction = float(
            self.model.predict(features)[0]
        )

        analytics["cost_per_room"] = (
            prediction /
            cleaned_data["total_rooms"]
        )

        # -------------------------------
        # Prediction Interval
        # -------------------------------

        interval = Utils.prediction_interval(
            self.model,
            features
        )

        # -------------------------------
        # Category
        # -------------------------------

        category = Utils.price_category(
            prediction
        )

        premium = Utils.is_premium(
            prediction
        )

        # -------------------------------
        # Luxury Score
        # -------------------------------

        luxury = Utils.luxury_score(

            predicted_price=prediction,

            median_income=cleaned_data[
                "median_income"
            ],

            rooms_per_household=analytics[
                "rooms_per_household"
            ],

            bedrooms_per_room=analytics[
                "bedrooms_per_room"
            ],

            ocean_proximity=cleaned_data[
                "ocean_proximity"
            ],
        )

        # -------------------------------
        # Summary
        # -------------------------------

        summary = Utils.build_summary(

            prediction=prediction,

            category=category,

            luxury_score=luxury,

            premium=premium,

            interval=interval,
        )

        return {

            "prediction": prediction,

            "formatted_price":
                Utils.format_currency(prediction),

            "category": category,

            "premium_house": premium,

            "luxury_score": luxury,

            "analytics": analytics,

            "prediction_interval": {

                "mean": interval[0],

                "lower": interval[1],

                "upper": interval[2],

                "std": interval[3],
            },

            "summary": summary,

            "features":

                self.feature_engineer.feature_summary(
                    features
                ),

            "input": cleaned_data,
        }

    # --------------------------------------------------

    def predict_price(self, input_data):

        """
        Returns only predicted price.
        """

        return self.predict(
            input_data
        )["prediction"]

    # --------------------------------------------------

    def predict_category(self, input_data):

        """
        Returns only price category.
        """

        return self.predict(
            input_data
        )["category"]

    # --------------------------------------------------

    def confidence_score(self, input_data):
        """
        Estimate confidence from
        Random Forest prediction variance.
        """

        result = self.predict(
            input_data
        )

        std = result[
            "prediction_interval"
        ]["std"]

        confidence = max(
            0,
            100 - std / 5000
        )

        return round(
            confidence,
            2
        )

    # --------------------------------------------------

    def model_information(self):

        """
        Returns metadata about model.
        """

        info = {

            "Model":

                self.model.__class__.__name__,

            "Number of Features":

                len(
                    self.model.feature_names_in_
                ),

            "Feature Names":

                list(
                    self.model.feature_names_in_
                ),

            "Supports Feature Importance":

                hasattr(
                    self.model,
                    "feature_importances_"
                ),

            "Supports Prediction Interval":

                hasattr(
                    self.model,
                    "estimators_"
                ),
        }

        return info

    # --------------------------------------------------

    def feature_importance(self):

        """
        Return feature importance
        if available.
        """

        return self.feature_engineer.feature_importance()

    # --------------------------------------------------

    def diagnostics(self, input_data):

        """
        Pipeline diagnostics.
        """

        cleaned = InputValidator.validate_and_clean(
            input_data
        )

        features = self.feature_engineer.transform(
            cleaned
        )

        diagnostics = self.feature_engineer.diagnostics(
            features
        )

        diagnostics["Model"] = self.model_name

        diagnostics["Model Path"] = str(
            self.model_path
        )

        return diagnostics

    # --------------------------------------------------

    def __repr__(self):

        return (

            f"HousingPricePredictor("
            f"model='{self.model_name}', "
            f"features={len(self.model.feature_names_in_)})"

        )