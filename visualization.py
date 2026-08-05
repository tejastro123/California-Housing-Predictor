"""
=========================================================
Visualization Module
California Housing Price Predictor
=========================================================

Creates interactive Folium maps for predictions.

Features
--------
✓ Multiple map layers
✓ Satellite view
✓ Dark mode
✓ Color-coded markers
✓ Circle markers
✓ FontAwesome icons
✓ Heatmap support
✓ Prediction popup
✓ Automatic HTML export
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import folium
from folium.plugins import HeatMap

from utils import Utils


class HousingVisualizer:
    """
    Interactive Folium visualization.
    """

    # --------------------------------------------------

    def __init__(self, zoom_start=10):

        self.zoom_start = zoom_start

    # --------------------------------------------------

    def create_prediction_map(
        self,
        result: dict,
        save_path="maps/Prediction_Map.html",
        dataset=None,
    ):
        """
        Create prediction map.

        Parameters
        ----------
        result : predictor output

        dataset : optional housing dataframe
                  used for heatmap
        """

        house = result["input"]

        price = result["prediction"]

        center = [
            house["latitude"],
            house["longitude"],
        ]

        # -----------------------------------------
        # Base Map
        # -----------------------------------------

        m = folium.Map(
            location=center,
            zoom_start=self.zoom_start,
            control_scale=True,
        )

        # -----------------------------------------
        # Map Layers
        # -----------------------------------------

        folium.TileLayer(
            "OpenStreetMap",
            name="OpenStreetMap",
        ).add_to(m)

        folium.TileLayer(
            "CartoDB Positron",
            name="Light",
        ).add_to(m)

        folium.TileLayer(
            "CartoDB Dark_Matter",
            name="Dark",
        ).add_to(m)

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
        ).add_to(m)

        # -----------------------------------------
        # Optional Heatmap
        # -----------------------------------------

        if dataset is not None:

            heat_data = dataset[
                [
                    "latitude",
                    "longitude",
                ]
            ].values.tolist()

            HeatMap(
                heat_data,
                radius=8,
            ).add_to(m)

        # -----------------------------------------
        # Marker Color
        # -----------------------------------------

        color = Utils.marker_color(price)

        # -----------------------------------------
        # Popup
        # -----------------------------------------

        popup_html = self._popup(result)

        # -----------------------------------------
        # Circle Marker
        # -----------------------------------------

        folium.CircleMarker(

            location=center,

            radius=12,

            color=color,

            fill=True,

            fill_color=color,

            fill_opacity=0.85,

            weight=3,

            popup=folium.Popup(
                popup_html,
                max_width=350,
            ),

            tooltip="Prediction",

        ).add_to(m)

        # -----------------------------------------
        # Home Icon
        # -----------------------------------------

        folium.Marker(

            location=center,

            icon=folium.Icon(

                color=color,

                icon="home",

                prefix="fa",

            ),

            tooltip="House",

        ).add_to(m)

        # -----------------------------------------
        # Layer Control
        # -----------------------------------------

        folium.LayerControl().add_to(m)

        # -----------------------------------------
        # Save
        # -----------------------------------------

        Path(save_path).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        m.save(save_path)

        return m

    # --------------------------------------------------

    @staticmethod
    def _popup(result):

        analytics = result["analytics"]

        interval = result["prediction_interval"]

        return f"""
        <div style="font-family:Arial;
                    width:280px;">

        <h3>
        California Housing Prediction
        </h3>

        <hr>

        <b>Predicted Price</b><br>

        {result["formatted_price"]}

        <br><br>

        <b>Category</b><br>

        {result["category"]}

        <br><br>

        <b>Luxury Score</b><br>

        {result["luxury_score"]}/100

        <br><br>

        <b>Premium House</b><br>

        {"YES" if result["premium_house"] else "NO"}

        <br><br>

        <b>Median Income</b><br>

        {result["input"]["median_income"]}

        <br><br>

        <b>Rooms</b><br>

        {result["input"]["total_rooms"]}

        <br><br>

        <b>Bedrooms</b><br>

        {result["input"]["total_bedrooms"]}

        <br><br>

        <b>Population</b><br>

        {result["input"]["population"]}

        <br><br>

        <b>Ocean</b><br>

        {result["input"]["ocean_proximity"]}

        <br><br>

        <b>Prediction Interval</b><br>

        {Utils.format_currency(interval["lower"])}
        -

        {Utils.format_currency(interval["upper"])}

        <br><br>

        <b>Rooms / Household</b><br>

        {analytics["rooms_per_household"]:.2f}

        <br>

        <b>Bedrooms / Room</b><br>

        {analytics["bedrooms_per_room"]:.3f}

        <br>

        <b>Population Density</b><br>

        {analytics["population_density"]:.2f}

        </div>
        """

    # --------------------------------------------------

    @staticmethod
    def save_map(map_object, filename):

        Path(filename).parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        map_object.save(filename)

    # --------------------------------------------------

    @staticmethod
    def open_browser(filename):
        """
        Optional helper.
        """

        import webbrowser

        webbrowser.open(
            Path(filename).absolute().as_uri()
        )

    # --------------------------------------------------

    @staticmethod
    def add_dataset_markers(
        map_object,
        dataset,
        limit=500,
    ):
        """
        Plot housing dataset.
        """

        subset = dataset.head(limit)

        for _, row in subset.iterrows():

            folium.CircleMarker(

                location=[
                    row["latitude"],
                    row["longitude"],
                ],

                radius=2,

                color="blue",

                fill=True,

                fill_opacity=0.3,

            ).add_to(map_object)

    # --------------------------------------------------

    @staticmethod
    def add_heatmap(
        map_object,
        dataset,
    ):
        """
        Add HeatMap separately.
        """

        HeatMap(

            dataset[
                [
                    "latitude",
                    "longitude",
                ]
            ].values.tolist(),

            radius=8,

        ).add_to(map_object)

    # --------------------------------------------------

    @staticmethod
    def map_statistics(dataset):

        return {

            "Total Districts": len(dataset),

            "Latitude Range": (
                dataset["latitude"].min(),
                dataset["latitude"].max(),
            ),

            "Longitude Range": (
                dataset["longitude"].min(),
                dataset["longitude"].max(),
            ),
        }

    # --------------------------------------------------

    def __repr__(self):

        return (
            "HousingVisualizer("
            f"zoom_start={self.zoom_start})"
        )