from steps.data_loader import load_data
from steps.data_preprocessor import preprocess_data
from steps.data_validator import validate_data
from steps.data_visualizer import visualize_sales_data
from steps.model_evaluator import evaluate_models
from steps.predictor import generate_forecasts
from zenml import get_pipeline_context, pipeline


@pipeline(name="retail_forecast_inference_pipeline")
def inference_pipeline():
    """Pipeline to make retail demand forecasts using trained Prophet models.

    This pipeline is for when you already have trained models and want to
    generate new forecasts without retraining.

    Steps:
    1. Load sales data
    2. Preprocess data
    3. Generate forecasts using provided models or simple baseline models

    Returns:
        combined_forecast: Combined dataframe with all series forecasts
        forecast_dashboard: HTML dashboard with forecast visualizations
        sales_visualization: Interactive visualization of historical sales patterns
    """
    # Load data
    sales_data = load_data()

    # Validate data quality and fix common issues
    sales_data_validated, _ = validate_data(
        sales_data=sales_data,
        calendar_data=sales_data  # Using sales data as calendar proxy for now
    )

    # Preprocess data
    train_data_dict, test_data_dict, series_ids = preprocess_data(
        sales_data=sales_data_validated,
        test_size=0.05,  # Just a small test set for visualization purposes
    )

    # Create interactive visualizations of historical sales patterns
    sales_viz = visualize_sales_data(
        sales_data=sales_data,
        train_data_dict=train_data_dict,
        test_data_dict=test_data_dict,
        series_ids=series_ids,
    )

    # Get the models from the Model Registry
    models = get_pipeline_context().model.get_artifact(
        "trained_prophet_models"
    )

    # Evaluate models on current data for quality control
    metrics, evaluation_report = evaluate_models(
        models=models, 
        test_data_dict=test_data_dict, 
        series_ids=series_ids
    )

    # Generate forecasts
    _, combined_forecast, forecast_dashboard = generate_forecasts(
        models=models,
        train_data_dict=train_data_dict,
        series_ids=series_ids,
    )

    # Return forecast data, dashboard, and quality metrics
    return combined_forecast, forecast_dashboard, sales_viz, metrics, evaluation_report
