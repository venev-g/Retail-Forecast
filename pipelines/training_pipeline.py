"""
Advanced retail forecasting pipeline with optimized Prophet features.
Combines traditional pipeline with hyperparameter optimization and enhanced features.
"""

import logging
from typing import Dict, Tuple

from steps.data_loader import load_data
from steps.data_preprocessor import preprocess_data
from steps.data_validator import validate_data
from steps.data_visualizer import visualize_sales_data
from steps.model_evaluator import evaluate_models
from steps.model_trainer import train_model
from steps.predictor import generate_forecasts
from typing_extensions import Annotated
from zenml import pipeline
from zenml.config import DockerSettings
from zenml.types import HTMLString

logger = logging.getLogger(__name__)

# Optimized Docker settings for the pipeline
docker_settings = DockerSettings(
    requirements=[
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "prophet>=1.1.5",
        "matplotlib>=3.5.0",
        "plotly>=5.0.0",
        "scikit-learn>=1.0.0",
        "pyarrow>=10.0.0",
        "fastparquet>=0.8.0",
        "typing_extensions>=4.0.0",
        "scipy>=1.9.0",
    ]
)


@pipeline(name="retail_forecast_pipeline", settings={"docker": docker_settings})
def training_pipeline(
    # Data preprocessing parameters (optimized defaults)
    test_size: float = 0.2,
    min_train_size: int = 50,
    outlier_method: str = "modified_zscore",
    outlier_threshold: float = 3.5,
    enable_advanced_features: bool = True,
    
    # Model optimization parameters
    enable_hyperparameter_tuning: bool = True,
    max_optimization_evals: int = 16,
    use_warm_start: bool = True,
    
    # Evaluation parameters
    calculate_residuals: bool = True,
    residual_analysis: bool = True,
) -> Tuple[
    Annotated[Dict[str, float], "model_metrics"],
    Annotated[HTMLString, "evaluation_report"],
    Annotated[HTMLString, "forecast_dashboard"],
    Annotated[HTMLString, "sales_visualization"],
]:
    """Advanced retail forecasting pipeline with optimized Prophet features.

    This pipeline combines traditional forecasting with advanced optimization techniques:
    
    Steps:
    1. Load sales data
    2. Validate data quality and fix common issues
    3. Advanced preprocessing with feature engineering and outlier detection
    4. Create interactive visualizations of historical sales patterns
    5. Train optimized Prophet models with hyperparameter tuning
    6. Comprehensive model evaluation with residual analysis
    7. Generate forecasts for future periods with confidence intervals

    Advanced Features:
    - Hyperparameter optimization via cross-validation
    - Advanced feature engineering (20+ retail-specific features)
    - Custom seasonalities (monthly, bi-weekly patterns)
    - Enhanced holiday modeling with retail calendar
    - Robust outlier detection and data quality improvements
    - Warm-start optimization for efficient training

    Args:
        test_size: Proportion of data to use for testing (default: 0.2)
        min_train_size: Minimum training samples required (default: 50)
        outlier_method: Method for outlier detection ('modified_zscore', 'iqr', 'zscore')
        outlier_threshold: Threshold for outlier detection (default: 3.5)
        enable_advanced_features: Whether to add advanced time and retail features
        enable_hyperparameter_tuning: Whether to optimize hyperparameters via cross-validation
        max_optimization_evals: Maximum hyperparameter combinations to evaluate
        use_warm_start: Whether to reuse hyperparameters across similar series
        calculate_residuals: Whether to calculate model residuals
        residual_analysis: Whether to perform detailed residual analysis

    Returns:
        model_metrics: Dictionary of comprehensive performance metrics
        evaluation_report: HTML report of model evaluation with residual analysis
        forecast_dashboard: Interactive HTML dashboard of forecasts
        sales_visualization: Interactive visualization of historical sales patterns
    """
    # Load synthetic retail data
    logger.info("Starting optimized retail forecasting pipeline")
    sales_data = load_data()

    # Validate data quality and fix common issues
    sales_data_validated, _ = validate_data(
        sales_data=sales_data,
        calendar_data=sales_data  # Using sales data as calendar proxy for now
    )

    # Advanced preprocessing with optimization parameters
    logger.info("Applying advanced data preprocessing with feature engineering")
    train_data_dict, test_data_dict, series_ids, holiday_df = preprocess_data(
        sales_data=sales_data_validated,
        test_size=test_size,
        min_train_size=min_train_size,
        outlier_method=outlier_method,
        outlier_threshold=outlier_threshold,
        enable_advanced_features=enable_advanced_features,
    )

    # Create interactive visualizations of historical sales patterns
    sales_viz = visualize_sales_data(
        sales_data=sales_data,
        train_data_dict=train_data_dict,
        test_data_dict=test_data_dict,
        series_ids=series_ids,
    )

    # Train optimized Prophet models with hyperparameter tuning
    logger.info("Training optimized models with hyperparameter tuning")
    models = train_model(
        train_data_dict=train_data_dict,
        series_ids=series_ids,
        holiday_dataframe=holiday_df,
        enable_hyperparameter_tuning=enable_hyperparameter_tuning,
        max_optimization_evals=max_optimization_evals,
        use_warm_start=use_warm_start,
    )

    # Comprehensive model evaluation with residual analysis
    logger.info("Evaluating optimized models with comprehensive metrics")
    metrics, evaluation_report = evaluate_models(
        models=models,
        test_data_dict=test_data_dict,
        series_ids=series_ids,  # Use the original series_ids
        metrics=["mae", "rmse", "mape", "smape"],
    )

    # Generate forecasts with optimized models
    logger.info("Generating forecasts with optimized models")
    _, _, forecast_dashboard = generate_forecasts(
        models=models,
        train_data_dict=train_data_dict,
        series_ids=series_ids,  # Use the original series_ids
    )

    logger.info("Optimized retail forecasting pipeline completed successfully")
    logger.info("Models trained with enhanced features and optimization")

    return metrics, evaluation_report, forecast_dashboard, sales_viz


if __name__ == "__main__":
    # Run the optimized pipeline with production settings
    optimized_pipeline = training_pipeline(
        # Enhanced preprocessing
        test_size=0.2,
        min_train_size=50,
        outlier_method="modified_zscore",
        outlier_threshold=3.5,
        enable_advanced_features=True,
        
        # Model optimization
        enable_hyperparameter_tuning=True,
        max_optimization_evals=16,  # Increased for better optimization
        use_warm_start=True,
        
        # Comprehensive evaluation
        calculate_residuals=True,
        residual_analysis=True,
    )
    
    metrics, evaluation_report, forecast_dashboard, sales_viz = optimized_pipeline()
    print(f"Pipeline completed successfully. Achieved MAE: {metrics.get('avg_mae', 'N/A')}")
    print("Comprehensive evaluation and forecasting dashboard generated.")
