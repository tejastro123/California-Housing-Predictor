"""
=========================================================
California Housing Price Predictor
Single-File Gradio Dashboard (gradio_app.py)
=========================================================

This is the main application file. It replaces app.py and includes:
  - Interactive Gradio web dashboard
  - Prediction with analytics & engineered features
  - Pipeline diagnostics
  - Feature importance
  - Console report
  - Map generation & report exports
"""

import os
import json
import pandas as pd
import numpy as np
import gradio as gr
import folium

from predictor import HousingPricePredictor
from visualization import HousingVisualizer
from report import PredictionReport
from utils import Utils

# 1. Initialize Predictor, Visualizer, Reporter
MODEL_PATH = "model/California_Housing_Price_Model.pkl"

if os.path.exists(MODEL_PATH):
    predictor = HousingPricePredictor(MODEL_PATH)
else:
    predictor = None

visualizer = HousingVisualizer()
reporter = PredictionReport()

# Default dataset for heatmap if available
DATASET_PATH = "data/housing.csv"
if os.path.exists(DATASET_PATH):
    try:
        housing_df = pd.read_csv(DATASET_PATH)
    except Exception:
        housing_df = None
else:
    housing_df = None


def predict_and_generate_outputs(
    longitude,
    latitude,
    housing_median_age,
    total_rooms,
    total_bedrooms,
    population,
    households,
    median_income,
    ocean_proximity,
):
    if predictor is None:
        raise RuntimeError("Model file not found at 'model/model.pkl'. Please place model in path.")

    input_data = {
        "longitude": float(longitude),
        "latitude": float(latitude),
        "housing_median_age": float(housing_median_age),
        "total_rooms": float(total_rooms),
        "total_bedrooms": float(total_bedrooms),
        "population": float(population),
        "households": float(households),
        "median_income": float(median_income),
        "ocean_proximity": str(ocean_proximity).upper(),
    }

    # Execute Prediction
    result = predictor.predict(input_data)

    # 1. Map Generation
    map_html_path = "maps/Prediction_Map.html"
    visualizer.create_prediction_map(
        result=result,
        save_path=map_html_path,
        dataset=housing_df,
    )

    with open(map_html_path, "r", encoding="utf-8") as f:
        map_html_content = f.read()

    # Escape quotes cleanly for iframe srcdoc
    escaped_map_content = map_html_content.replace('"', '&quot;')
    map_iframe = f'<iframe srcdoc="{escaped_map_content}" width="100%" height="500px" style="border:none; border-radius:10px;"></iframe>'

    # 2. Export Reports
    reporter.export_all(result)

    pdf_path = "reports/Prediction_Report.pdf"
    json_path = "reports/Prediction_Report.json"
    csv_path = "reports/Prediction_Report.csv"

    # Summary Card Markdown / Metrics
    pred_formatted = result["formatted_price"]
    category = result["category"]
    luxury_score = f"{result['luxury_score']} / 100"
    premium_house = "✅ YES — Premium House" if result.get("premium_house") else "❌ NO"
    interval = result["prediction_interval"]
    confidence_interval_str = f"{Utils.format_currency(interval['lower'])} - {Utils.format_currency(interval['upper'])}"

    # Engineered Analytics Features (from app.py)
    analytics = result.get("analytics", {})
    analytics_md = "### 🔧 Engineered Features\n"
    if analytics:
        analytics_md += f"* **Rooms / Household:** `{analytics.get('rooms_per_household', 'N/A'):.3f}`\n"
        analytics_md += f"* **Bedrooms / Room:** `{analytics.get('bedrooms_per_room', 'N/A'):.3f}`\n"
        analytics_md += f"* **Population Density:** `{analytics.get('population_density', 'N/A'):.3f}`\n"
        analytics_md += f"* **Cost Per Room:** `${analytics.get('cost_per_room', 0):,.2f}`\n"
    else:
        analytics_md += "_Analytics not available._\n"

    # Pipeline Diagnostics (from app.py)
    diagnostics = predictor.diagnostics(input_data)
    diagnostics_md = "### 🔍 Pipeline Diagnostics\n"
    for k, v in diagnostics.items():
        diagnostics_md += f"* **{k}:** `{v}`\n"

    # Console Report (from app.py)
    console_report_str = reporter.console_report(result)

    summary_md = (
        "### 📊 Prediction Results\n"
        f"* **Estimated Price:** <span style='font-size: 1.5em; color: #5d5448; font-weight: bold;'>{pred_formatted}</span>\n"
        f"* **Category:** `{category}`\n"
        f"* **Luxury Score:** `{luxury_score}`\n"
        f"* **Premium House:** {premium_house}\n"
        f"* **95% Confidence Interval:** `{confidence_interval_str}`\n"
        f"* **Standard Deviation:** `${interval['std']:,.2f}`\n"
    )

    return (
        summary_md,
        pred_formatted,
        category,
        luxury_score,
        confidence_interval_str,
        premium_house,
        analytics_md,
        diagnostics_md,
        console_report_str,
        map_iframe,
        pdf_path,
        json_path,
        csv_path,
        map_html_path,
    )


def get_feature_importance_df():
    if predictor and hasattr(predictor, "feature_importance"):
        fi = predictor.feature_importance()
        if fi is not None:
            return fi
    return pd.DataFrame({"Feature": [], "Importance": []})


def get_model_info_md():
    if predictor is None:
        return "Model not loaded."
    info = predictor.model_information()
    md = "### 🤖 Model Specification\n"
    for k, v in info.items():
        md += f"* **{k}:** `{v}`\n"
    return md


def run_cli_prediction():
    """Replicates app.py main() logic for the Gradio CLI tab."""
    if predictor is None:
        return "❌ Model not loaded. Cannot run CLI prediction."
    default_house = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41,
        "total_rooms": 880,
        "total_bedrooms": 129,
        "population": 322,
        "households": 126,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
    }
    result = predictor.predict(default_house)
    interval = result["prediction_interval"]
    analytics = result.get("analytics", {})
    diagnostics = predictor.diagnostics(default_house)
    importance = predictor.feature_importance()
    report_text = reporter.console_report(result)

    output = []
    output.append("=" * 70)
    output.append("California Housing Price Predictor — CLI Demo (Default Example)")
    output.append("=" * 70)
    output.append(f"Predicted Price : {result['formatted_price']}")
    output.append(f"Category        : {result['category']}")
    output.append(f"Luxury Score    : {result['luxury_score']}/100")
    output.append(f"Premium House   : {'YES' if result.get('premium_house') else 'NO'}")
    output.append(f"Interval (95%)  : ${interval['lower']:,.2f}  -  ${interval['upper']:,.2f}")
    output.append("")
    output.append("Engineered Features")
    output.append("-" * 40)
    if analytics:
        output.append(f"  Rooms / Household : {analytics.get('rooms_per_household', 'N/A'):.3f}")
        output.append(f"  Bedrooms / Room   : {analytics.get('bedrooms_per_room', 'N/A'):.3f}")
        output.append(f"  Population Density: {analytics.get('population_density', 'N/A'):.3f}")
        output.append(f"  Cost Per Room     : ${analytics.get('cost_per_room', 0):,.2f}")
    output.append("")
    output.append("Pipeline Diagnostics")
    output.append("-" * 40)
    for k, v in diagnostics.items():
        output.append(f"  {k}: {v}")
    output.append("")
    if importance is not None:
        output.append("Top 10 Most Important Features")
        output.append("-" * 40)
        output.append(importance.head(10).to_string(index=False))
        output.append("")
    output.append("=" * 70)
    output.append("Prediction Report")
    output.append("=" * 70)
    output.append(report_text)
    output.append("=" * 70)
    output.append("Prediction Complete.")
    output.append("=" * 70)
    return "\n".join(output)


# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Retro-Vintage UI — California Housing Predictor
# ─────────────────────────────────────────────
custom_css = '''
/* Global Reset */
html, body, .gradio-container, grad-app, .dark, [class*="dark"] {
    background-color: #e0dacf !important;
    background: #e0dacf !important;
    font-family: 'Courier New', Courier, monospace !important;
    color: #000000 !important;
}

/* Force black text everywhere */
*, 
span, label, input, select, textarea, button, p, h1, h2, h3, h4, h5, h6,
.gr-markdown, .gr-markdown *, .markdown, .markdown *, div, td, th {
    font-family: 'Courier New', Courier, monospace !important;
    color: #000000 !important;
}

/* Base reset for inner elements - NO nested borders or shadows */
*, *:before, *:after {
    box-sizing: border-box;
}

/* Remove default borders and shadows from inner component wrappers */
div, fieldset, form, label, .gr-form, .gr-box, .gr-panel, div[class*="form"], div[class*="block"], table, th, td, tr, thead, tbody, pre, code, span, p, h1, h2, h3, h4, h5, h6 {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* Single Outer Box Styling for Group Cards */
.single-card, div.group, .gr-group, fieldset.gr-group {
    background-color: #e0dacf !important;
    background: #e0dacf !important;
    border: 3px solid #5d5448 !important;
    border-radius: 8px !important;
    box-shadow: 6px 6px 0px #5d5448, -2px -2px 0px #fff !important;
    margin-bottom: 16px !important;
    padding: 16px !important;
}

/* Specific styling for sliders, number fields, dropdowns and text inputs */
input[type="text"], input[type="number"], select, textarea, .gr-input input, input {
    background-color: #f3ede2 !important;
    background: #f3ede2 !important;
    border: 2px solid #5d5448 !important;
    border-radius: 4px !important;
    color: #1a1612 !important;
    box-shadow: inset 2px 2px 5px rgba(93, 84, 72, 0.2) !important;
    font-weight: bold !important;
}

/* Retro Buttons */
button, button.primary, .gr-button-primary {
    background-color: #c8bda7 !important;
    background: #c8bda7 !important;
    color: #000000 !important;
    border: 3px solid #5d5448 !important;
    border-radius: 6px !important;
    box-shadow: 4px 4px 0px #5d5448 !important;
    font-weight: 800 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    cursor: pointer !important;
    transition: all 0.1s ease !important;
}

button:hover, button.primary:hover, .gr-button-primary:hover {
    background-color: #b5aa93 !important;
    background: #b5aa93 !important;
    box-shadow: 2px 2px 0px #5d5448 !important;
    transform: translate(2px, 2px) !important;
}

button:active, button.primary:active, .gr-button-primary:active {
    box-shadow: 0px 0px 0px #5d5448 !important;
    transform: translate(4px, 4px) !important;
}

/* Tab Styling */
.tabs {
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
.tab-nav button, .tabs button {
    background-color: #f3ede2 !important;
    background: #f3ede2 !important;
    border: 2px solid #5d5448 !important;
    border-radius: 4px 4px 0px 0px !important;
    margin-right: 4px !important;
    color: #000000 !important;
}
.tab-nav button.selected, .tabs button.selected {
    background-color: #c8bda7 !important;
    background: #c8bda7 !important;
    border-bottom: 2px solid #c8bda7 !important;
    font-weight: bold !important;
    color: #000000 !important;
}

footer {
    visibility: hidden !important;
    display: none !important;
}

.divider {
  height: 2px;
  background: #5d5448;
  margin: 20px 0;
}
'''

HEADER_HTML = '''
<div style="
  background-color: #e0dacf;
  border: 4px solid #5d5448;
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 6px 6px 0px #5d5448;
  display: flex;
  align-items: center;
  gap: 20px;
">
  <div style="font-size: 2.5em; line-height:1;">🏡</div>
  <div style="flex:1;">
    <div style="
      font-family: 'Courier New', Courier, monospace;
      font-size: 1.8rem;
      font-weight: 800;
      color: #231f1a;
      text-transform: uppercase;
      letter-spacing: 1px;
    ">California Housing Price Predictor</div>
    <div style="
      font-family: 'Courier New', Courier, monospace;
      font-size: 0.9rem;
      color: #5d5448;
      margin-top: 4px;
      font-weight: bold;
    ">ML-POWERED REAL ESTATE ANALYTICS & DASHBOARD (RETRO EDITION)</div>
  </div>
</div>
'''

with gr.Blocks(title="California Housing Price Predictor", css=custom_css, theme=gr.themes.Default()) as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs(elem_classes=["tabs"]):
        with gr.TabItem("Dashboard"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["single-card"]):
                    gr.Markdown("### Property Parameters")
                    
                    ocean_proximity = gr.Dropdown(
                        choices=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"],
                        value="NEAR BAY",
                        label="Ocean Proximity",
                    )
                    latitude = gr.Slider(minimum=32.5, maximum=42.0, value=37.88, step=0.01, label="Latitude")
                    longitude = gr.Slider(minimum=-124.5, maximum=-114.0, value=-122.23, step=0.01, label="Longitude")
                    median_income = gr.Slider(minimum=0.5, maximum=15.0, value=8.32, step=0.1, label="Median Income (x$10k)")
                    housing_median_age = gr.Slider(minimum=1, maximum=52, value=41, step=1, label="Housing Median Age (years)")
                    
                    total_rooms = gr.Number(value=880, label="Total Rooms", precision=0)
                    total_bedrooms = gr.Number(value=129, label="Total Bedrooms", precision=0)
                    population = gr.Number(value=322, label="Population", precision=0)
                    households = gr.Number(value=126, label="Households", precision=0)

                    predict_btn = gr.Button("Predict Price", variant="primary")

                with gr.Column(scale=2, elem_classes=["single-card"]):
                    gr.Markdown("### Prediction Results")
                    
                    with gr.Row():
                        out_price = gr.Textbox(label="Estimated Price", interactive=False)
                        out_category = gr.Textbox(label="Category", interactive=False)
                        
                    with gr.Row():
                        out_luxury = gr.Textbox(label="Luxury Score", interactive=False)
                        out_interval = gr.Textbox(label="95% Confidence Interval", interactive=False)
                        
                    out_premium = gr.Textbox(label="Premium House", interactive=False)
                    
                    with gr.Accordion("Full Summary", open=True):
                        out_summary_md = gr.Markdown("_Fill in parameters and click **Predict Price** to get results._")
                        
                    with gr.Accordion("Engineered Features", open=False):
                        out_analytics_md = gr.Markdown("_Run prediction to view derived analytics._")
                        
                    with gr.Accordion("Pipeline Diagnostics", open=False):
                        out_diagnostics_md = gr.Markdown("_Run prediction to view pipeline diagnostics._")
                        
                    with gr.Accordion("Console Report", open=False):
                        out_console_report = gr.Textbox(label="Report Output", interactive=False, lines=12)
                        
                    with gr.Accordion("Interactive Spatial Map", open=False):
                        map_output = gr.HTML()
                        
                    with gr.Accordion("Export Reports", open=False):
                        with gr.Row():
                            btn_pdf = gr.File(label="PDF Report")
                            btn_json = gr.File(label="JSON Report")
                        with gr.Row():
                            btn_csv = gr.File(label="CSV Report")
                            btn_map = gr.File(label="HTML Map")

            predict_btn.click(
                fn=predict_and_generate_outputs,
                inputs=[longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity],
                outputs=[out_summary_md, out_price, out_category, out_luxury, out_interval, out_premium, out_analytics_md, out_diagnostics_md, out_console_report, map_output, btn_pdf, btn_json, btn_csv, btn_map],
            )

        with gr.TabItem("Feature Importance"):
            with gr.Column(elem_classes=["single-card"]):
                gr.Markdown("### Model Feature Importances\nRelative weights assigned by the XGBoost model to each engineered feature.")
                feature_imp_df = gr.Dataframe(value=get_feature_importance_df(), interactive=False)

        with gr.TabItem("Model Info"):
            with gr.Column(elem_classes=["single-card"]):
                gr.Markdown(get_model_info_md())

        with gr.TabItem("CLI Demo"):
            with gr.Column(elem_classes=["single-card"]):
                gr.Markdown("### CLI Demo\nRuns the same pipeline as app.py with the default San Francisco Bay Area example.")
                cli_run_btn = gr.Button("Run CLI Demo", variant="primary")
                cli_output = gr.Textbox(label="Terminal Output", interactive=False, lines=32)
                cli_run_btn.click(fn=run_cli_prediction, inputs=[], outputs=[cli_output])

        with gr.TabItem("About"):
            with gr.Column(elem_classes=["single-card"]):
                gr.Markdown("### About\nCalifornia Housing Price Predictor - Retro Edition.\nBITS F464 Machine Learning — Assignment 1.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)
