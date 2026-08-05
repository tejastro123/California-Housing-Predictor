"""
=========================================================
Input Validation Module
California Housing Price Predictor
=========================================================

This module validates user input before prediction.

Features
--------
✓ Required field checking
✓ Data type validation
✓ Missing value detection
✓ Range validation
✓ Ocean category validation
✓ Descriptive error messages
"""

from typing import Dict, Any


class ValidationError(Exception):
    """
    Custom exception raised when validation fails.
    """
    pass


class InputValidator:
    """
    Validates housing prediction input.
    """

    REQUIRED_FIELDS = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity",
    ]

    OCEAN_CATEGORIES = {
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN",
    }

    # California Dataset Bounds

    LONGITUDE_RANGE = (-124.35, -114.31)
    LATITUDE_RANGE = (32.54, 41.95)

    AGE_RANGE = (1, 100)

    POSITIVE_FIELDS = [
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]

    NUMERIC_FIELDS = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]

    # -----------------------------------------------------

    @classmethod
    def validate(cls, data: Dict[str, Any]) -> None:
        """
        Run all validations.

        Raises
        ------
        ValidationError
        """

        cls._check_required_fields(data)

        cls._check_missing_values(data)

        cls._check_numeric_fields(data)

        cls._check_ranges(data)

        cls._check_positive_values(data)

        cls._check_ocean_category(data)

        cls._logical_validation(data)

    # -----------------------------------------------------

    @classmethod
    def _check_required_fields(cls, data):

        missing = []

        for field in cls.REQUIRED_FIELDS:
            if field not in data:
                missing.append(field)

        if missing:
            raise ValidationError(
                f"Missing required fields: {', '.join(missing)}"
            )

    # -----------------------------------------------------

    @staticmethod
    def _check_missing_values(data):

        for key, value in data.items():

            if value is None:
                raise ValidationError(
                    f"{key} cannot be None."
                )

            if isinstance(value, str):

                if value.strip() == "":
                    raise ValidationError(
                        f"{key} cannot be empty."
                    )

    # -----------------------------------------------------

    @classmethod
    def _check_numeric_fields(cls, data):

        for field in cls.NUMERIC_FIELDS:

            try:
                float(data[field])

            except (ValueError, TypeError):

                raise ValidationError(
                    f"{field} must be numeric."
                )

    # -----------------------------------------------------

    @classmethod
    def _check_ranges(cls, data):

        lon = float(data["longitude"])

        lat = float(data["latitude"])

        age = float(data["housing_median_age"])

        if not (
            cls.LONGITUDE_RANGE[0]
            <= lon
            <= cls.LONGITUDE_RANGE[1]
        ):
            raise ValidationError(
                f"Longitude must be between "
                f"{cls.LONGITUDE_RANGE[0]} "
                f"and {cls.LONGITUDE_RANGE[1]}"
            )

        if not (
            cls.LATITUDE_RANGE[0]
            <= lat
            <= cls.LATITUDE_RANGE[1]
        ):
            raise ValidationError(
                f"Latitude must be between "
                f"{cls.LATITUDE_RANGE[0]} "
                f"and {cls.LATITUDE_RANGE[1]}"
            )

        if not (
            cls.AGE_RANGE[0]
            <= age
            <= cls.AGE_RANGE[1]
        ):
            raise ValidationError(
                f"Housing age must be between "
                f"{cls.AGE_RANGE[0]} and "
                f"{cls.AGE_RANGE[1]} years."
            )

    # -----------------------------------------------------

    @classmethod
    def _check_positive_values(cls, data):

        for field in cls.POSITIVE_FIELDS:

            value = float(data[field])

            if value <= 0:

                raise ValidationError(
                    f"{field} must be greater than zero."
                )

    # -----------------------------------------------------

    @classmethod
    def _check_ocean_category(cls, data):

        category = str(data["ocean_proximity"]).upper()

        if category not in cls.OCEAN_CATEGORIES:

            raise ValidationError(
                f"Invalid ocean proximity '{category}'. "
                f"Allowed values: "
                f"{', '.join(sorted(cls.OCEAN_CATEGORIES))}"
            )

    # -----------------------------------------------------

    @staticmethod
    def _logical_validation(data):
        """
        Logical consistency checks.
        """

        rooms = float(data["total_rooms"])

        bedrooms = float(data["total_bedrooms"])

        population = float(data["population"])

        households = float(data["households"])

        if bedrooms > rooms:

            raise ValidationError(
                "Bedrooms cannot exceed total rooms."
            )

        if households > population:

            raise ValidationError(
                "Households cannot exceed population."
            )

        if rooms / households > 100:

            raise ValidationError(
                "Rooms per household is unrealistically high."
            )

        if population / households > 30:

            raise ValidationError(
                "Population per household is unrealistically high."
            )

    # -----------------------------------------------------

    @classmethod
    def sanitize(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Return cleaned copy of input.
        """

        cleaned = data.copy()

        for field in cls.NUMERIC_FIELDS:

            cleaned[field] = float(cleaned[field])

        cleaned["housing_median_age"] = int(
            cleaned["housing_median_age"]
        )

        cleaned["ocean_proximity"] = (
            str(cleaned["ocean_proximity"])
            .strip()
            .upper()
        )

        return cleaned

    # -----------------------------------------------------

    @classmethod
    def validate_and_clean(cls, data):
        """
        Main public method.

        Returns
        -------
        Clean validated dictionary.
        """

        cls.validate(data)

        return cls.sanitize(data)