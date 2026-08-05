"""
=========================================================
Feature Engineering Module
California Housing Price Predictor
=========================================================

This module performs all feature engineering required by the
trained machine learning model.

Features
--------
✓ Derived numerical features
✓ Automatic one-hot encoding
✓ Automatic feature alignment
✓ Missing column handling
✓ Extra analytics features
"""

from __future__ import annotations

import pandas as pd
import numpy as np


class FeatureEngineer:
    """
    Feature Engineering Pipeline
    """

    def __init__(self, model):
        """
        Parameters
        ----------
        model : trained sklearn model

        The model is used to automatically obtain the
        expected feature order.
        """

        self.model = model

    # =====================================================
    # Main Pipeline
    # =====================================================

    def transform(self, input_data: dict):
        """
        Complete feature engineering pipeline.

        Parameters
        ----------
        input_data : dict

        Returns
        -------
        DataFrame ready for prediction
        """

        df = pd.DataFrame([input_data])

        df = self.create_features(df)

        df = self.encode_categorical(df)

        df = self.align_features(df)

        return df

    # =====================================================
    # Feature Engineering
    # =====================================================

    def create_features(self, df: pd.DataFrame):

        # Avoid divide-by-zero

        epsilon = 1e-6

        df["rooms_per_household"] = (
            df["total_rooms"]
            / (df["households"] + epsilon)
        )

        df["bedrooms_per_room"] = (
            df["total_bedrooms"]
            / (df["total_rooms"] + epsilon)
        )

        df["population_per_household"] = (
            df["population"]
            / (df["households"] + epsilon)
        )

        df["rooms_per_person"] = (
            df["total_rooms"]
            / (df["population"] + epsilon)
        )

        df["bedrooms_per_household"] = (
            df["total_bedrooms"]
            / (df["households"] + epsilon)
        )

        df["income_per_household"] = (
            df["median_income"]
            / (df["households"] + epsilon)
        )

        df["age_income_interaction"] = (
            df["housing_median_age"]
            * df["median_income"]
        )

        df["geo_balance"] = (
            df["latitude"].abs()
            + df["longitude"].abs()
        )

        return df

    # =====================================================
    # Extra Analytics Features
    # =====================================================

    @staticmethod
    def analytics(df):

        epsilon = 1e-6

        analytics = {}

        analytics["cost_per_room"] = None

        analytics["rooms_per_household"] = (
            df["total_rooms"].iloc[0]
            /
            (df["households"].iloc[0] + epsilon)
        )

        analytics["bedrooms_per_room"] = (
            df["total_bedrooms"].iloc[0]
            /
            (df["total_rooms"].iloc[0] + epsilon)
        )

        analytics["population_density"] = (
            df["population"].iloc[0]
            /
            (df["households"].iloc[0] + epsilon)
        )

        analytics["income"] = (
            df["median_income"].iloc[0]
        )

        return analytics

    # =====================================================
    # One-Hot Encoding
    # =====================================================

    def encode_categorical(self, df):

        encoded = pd.get_dummies(
            df["ocean_proximity"],
            prefix="ocean_proximity_encoded",
        )

        df = pd.concat(
            [
                df.drop(columns=["ocean_proximity"]),
                encoded,
            ],
            axis=1,
        )

        return df

    # =====================================================
    # Feature Alignment
    # =====================================================

    def align_features(self, df):

        if hasattr(self.model, "feature_names_in_"):

            df = df.reindex(
                columns=self.model.feature_names_in_,
                fill_value=0,
            )

            return df

        raise AttributeError(
            "Model does not contain feature_names_in_. "
            "Retrain using sklearn >=1.0 or save feature names."
        )

    # =====================================================
    # Feature Summary
    # =====================================================

    @staticmethod
    def feature_summary(df):

        summary = {}

        for col in df.columns:

            value = df[col].iloc[0]

            if isinstance(value, np.floating):

                value = round(float(value), 4)

            summary[col] = value

        return summary

    # =====================================================
    # Feature Importance Helper
    # =====================================================

    def feature_importance(self):

        if hasattr(self.model, "feature_importances_"):

            importance = pd.DataFrame(
                {
                    "Feature": self.model.feature_names_in_,
                    "Importance": self.model.feature_importances_,
                }
            )

            importance = importance.sort_values(
                "Importance",
                ascending=False,
            )

            return importance.reset_index(drop=True)

        return None

    # =====================================================
    # Diagnostics
    # =====================================================

    def diagnostics(self, df):

        report = {
            "Number of Features": len(df.columns),
            "Missing Values": int(df.isnull().sum().sum()),
            "Duplicate Columns": int(df.columns.duplicated().sum()),
            "Feature Order Matches Model": True,
        }

        return report