"""
=========================================================
Report Generation Module
California Housing Price Predictor
=========================================================

Creates professional prediction reports.

Features
--------
✓ Console Report
✓ PDF Report
✓ JSON Export
✓ CSV Export
✓ Prediction Summary
✓ Engineered Features
✓ Prediction Interval
✓ Map Embedding
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)

from utils import Utils


class PredictionReport:

    """
    Generate prediction reports.
    """

    # =====================================================

    def __init__(self):

        self.styles = getSampleStyleSheet()

    # =====================================================

    def console_report(self, result):

        analytics = result["analytics"]

        interval = result["prediction_interval"]

        report = f"""

============================================================
          California Housing Prediction Report
============================================================

Prediction
----------

Predicted Price
    {result["formatted_price"]}

Category
    {result["category"]}

Luxury Score
    {result["luxury_score"]}/100

Premium House
    {"YES" if result["premium_house"] else "NO"}

Prediction Interval (95%)

    {Utils.format_currency(interval["lower"])}
        -
    {Utils.format_currency(interval["upper"])}

Model Statistics
----------------

Standard Deviation

    {interval["std"]:.2f}

Location
--------

Latitude

    {result["input"]["latitude"]}

Longitude

    {result["input"]["longitude"]}

House Information
-----------------

Median Income

    {result["input"]["median_income"]}

Rooms

    {result["input"]["total_rooms"]}

Bedrooms

    {result["input"]["total_bedrooms"]}

Population

    {result["input"]["population"]}

Households

    {result["input"]["households"]}

Ocean Proximity

    {result["input"]["ocean_proximity"]}

Engineered Features
-------------------

Rooms / Household

    {analytics["rooms_per_household"]:.3f}

Bedrooms / Room

    {analytics["bedrooms_per_room"]:.3f}

Population Density

    {analytics["population_density"]:.3f}

Cost / Room

    {Utils.format_currency(
        analytics["cost_per_room"]
    )}

Price Gauge

    {Utils.price_gauge(result["prediction"])}

Generated

    {datetime.now()}

============================================================
"""

        return report

    # =====================================================

    def save_console_report(
        self,
        result,
        filename="reports/Prediction_Report.txt",
    ):

        Path(filename).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                self.console_report(result)
            )

    # =====================================================

    def save_json(
        self,
        result,
        filename="reports/Prediction_Report.json",
    ):

        Path(filename).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                result,
                file,
                indent=4,
                default=str,
            )

    # =====================================================

    def save_csv(
        self,
        result,
        filename="reports/Prediction_Report.csv",
    ):

        flat = {}

        flat.update(result["input"])

        flat.update(result["analytics"])

        flat["Prediction"] = result["prediction"]

        flat["Category"] = result["category"]

        flat["Luxury Score"] = result[
            "luxury_score"
        ]

        flat["Premium House"] = result[
            "premium_house"
        ]

        df = pd.DataFrame([flat])

        Path(filename).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            filename,
            index=False,
        )

    # =====================================================

    def save_pdf(
        self,
        result,
        filename="reports/Prediction_Report.pdf",
        map_image=None,
    ):

        Path(filename).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        doc = SimpleDocTemplate(filename)

        story = []

        style = self.styles["Heading1"]

        story.append(
            Paragraph(
                "California Housing Prediction Report",
                style,
            )
        )

        story.append(
            Spacer(
                1,
                0.25 * inch,
            )
        )

        info = [

            ["Predicted Price",
             result["formatted_price"]],

            ["Category",
             result["category"]],

            ["Luxury Score",
             f'{result["luxury_score"]}/100'],

            ["Premium House",
             "YES" if result["premium_house"] else "NO"],

            ["Latitude",
             result["input"]["latitude"]],

            ["Longitude",
             result["input"]["longitude"]],

            ["Median Income",
             result["input"]["median_income"]],

            ["Ocean",
             result["input"]["ocean_proximity"]],
        ]

        table = Table(info)

        table.setStyle(

            TableStyle(

                [

                    ("GRID",
                     (0, 0),
                     (-1, -1),
                     1,
                     colors.black),

                    ("BACKGROUND",
                     (0, 0),
                     (0, -1),
                     colors.lightgrey),

                    ("FONTNAME",
                     (0, 0),
                     (-1, -1),
                     "Helvetica"),

                    ("BOTTOMPADDING",
                     (0, 0),
                     (-1, -1),
                     8),

                ]

            )

        )

        story.append(table)

        story.append(
            Spacer(
                1,
                0.3 * inch,
            )
        )

        interval = result["prediction_interval"]

        story.append(

            Paragraph(

                f"""
                <b>Prediction Interval (95%)</b><br/>

                {Utils.format_currency(interval["lower"])}
                -

                {Utils.format_currency(interval["upper"])}
                """,

                self.styles["BodyText"],

            )

        )

        story.append(
            Spacer(
                1,
                0.25 * inch,
            )
        )

        analytics = result["analytics"]

        story.append(

            Paragraph(

                "<b>Engineered Features</b>",

                self.styles["Heading2"],

            )

        )

        for key, value in analytics.items():

            story.append(

                Paragraph(

                    f"{key}: {value}",

                    self.styles["BodyText"],

                )

            )

        story.append(
            Spacer(
                1,
                0.25 * inch,
            )
        )

        if map_image:

            if Path(map_image).exists():

                img = Image(
                    map_image,
                    width=6 * inch,
                    height=4 * inch,
                )

                story.append(img)

        story.append(
            Spacer(
                1,
                0.2 * inch,
            )
        )

        story.append(

            Paragraph(

                f"""
Generated at

{datetime.now()}

California Housing Price Predictor
                """,

                self.styles["BodyText"],

            )

        )

        doc.build(story)

    # =====================================================

    def export_all(
        self,
        result,
        map_image=None,
    ):

        self.save_console_report(result)

        self.save_json(result)

        self.save_csv(result)

        self.save_pdf(
            result,
            map_image=map_image,
        )

    # =====================================================

    def summary(self, result):

        return {

            "Prediction":
                result["formatted_price"],

            "Category":
                result["category"],

            "Luxury Score":
                result["luxury_score"],

            "Premium":
                result["premium_house"],

            "Generated":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
        }

    # =====================================================

    def __repr__(self):

        return "PredictionReport()"