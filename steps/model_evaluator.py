import base64
import logging
from io import BytesIO
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from typing_extensions import Annotated
from zenml import log_metadata, step
from zenml.types import HTMLString

logger = logging.getLogger(__name__)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Symmetric Mean Absolute Percentage Error."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    return 100 * np.mean(np.abs(y_true - y_pred) / np.maximum(denominator, 1e-8))


def calculate_mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
    """Calculate Mean Absolute Scaled Error."""
    # Calculate naive forecast error (seasonal naive for daily data)
    naive_error = np.mean(np.abs(np.diff(y_train)))
    if naive_error == 0:
        naive_error = 1e-8
    
    mae = np.mean(np.abs(y_true - y_pred))
    return mae / naive_error


@step
def evaluate_models(
    models: Dict[str, Prophet],
    test_data_dict: Dict[str, pd.DataFrame],
    series_ids: List[str],
    forecast_horizon: int = 7, 
    metrics: List[str] = ["mae", "rmse", "mape", "smape"],
    cross_validation_periods: int = 5,
    horizon_days: int = 7,
) -> Tuple[
    Annotated[Dict[str, float], "performance_metrics"],
    Annotated[HTMLString, "evaluation_report"],
]:
    """Evaluate Prophet models with comprehensive metrics and validation.

    Args:
        models: Dictionary of trained Prophet models
        test_data_dict: Dictionary of test data for each series
        series_ids: List of series identifiers
        forecast_horizon: Number of periods to forecast
        metrics: List of metrics to calculate
        cross_validation_periods: Number of CV periods
        horizon_days: Horizon for cross-validation

    Returns:
        performance_metrics: Dictionary of average metrics across all series
        evaluation_report: HTML report with evaluation metrics and visualizations
    """
    # Initialize comprehensive metrics storage
    all_metrics = {metric: [] for metric in metrics}
    if "mase" in metrics:
        all_metrics["mase"] = []

    series_metrics = {}

    # Create a figure for plotting forecasts
    plt.figure(figsize=(12, len(series_ids) * 4))

    for i, series_id in enumerate(series_ids):
        if series_id not in models:
            logger.warning(f"No model found for {series_id}, skipping evaluation")
            continue
            
        logger.info(f"Evaluating model for {series_id}...")
        model = models[series_id]
        test_data = test_data_dict[series_id]

        # Debug: Check that test data exists
        logger.info(f"Test data shape for {series_id}: {test_data.shape}")
        logger.info(f"Test data date range: {test_data['ds'].min()} to {test_data['ds'].max()}")

        # Prepare test data for prediction
        test_future = test_data[['ds']].copy()
        
        # Handle additional regressors if they exist
        regressor_cols = ['is_weekend', 'is_month_end', 'is_month_start']
        for col in regressor_cols:
            if col in test_data.columns:
                test_future[col] = test_data[col]
        
        # Handle cap and floor for logistic growth
        if model.growth == 'logistic' and 'cap' in test_data.columns:
            test_future['cap'] = test_data['cap']
        if 'floor' in test_data.columns:
            test_future['floor'] = test_data['floor']

        try:
            # Make predictions for test dates
            forecast = model.predict(test_future)
            
            logger.info(f"Forecast shape: {forecast.shape}")
            logger.info(f"Forecast date range: {forecast['ds'].min()} to {forecast['ds'].max()}")

            # Ensure non-negative predictions for retail data
            forecast['yhat'] = np.maximum(forecast['yhat'], 0)
            forecast['yhat_lower'] = np.maximum(forecast['yhat_lower'], 0)
            forecast['yhat_upper'] = np.maximum(forecast['yhat_upper'], 0)

            # Merge forecasts with test data
            merged_data = pd.merge(
                test_data[['ds', 'y']],
                forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]],
                on="ds",
                how="inner",
            )

            logger.info(f"Merged data shape: {merged_data.shape}")
            if merged_data.empty:
                logger.warning(f"No matching dates between test data and forecast for {series_id}")
                continue
                
        except Exception as e:
            logger.error(f"Prediction failed for {series_id}: {str(e)}")
            continue

        # Calculate comprehensive metrics
        if len(merged_data) > 0:
            actuals = merged_data["y"].values
            predictions = merged_data["yhat"].values

            logger.info(f"Actuals range: {actuals.min()} to {actuals.max()}")
            logger.info(f"Predictions range: {predictions.min()} to {predictions.max()}")

            # Calculate basic metrics
            series_metrics_dict = {}
            
            if "mae" in metrics:
                mae = np.mean(np.abs(actuals - predictions))
                series_metrics_dict["mae"] = mae
                all_metrics["mae"].append(mae)
            
            if "rmse" in metrics:
                rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
                series_metrics_dict["rmse"] = rmse
                all_metrics["rmse"].append(rmse)
            
            if "mape" in metrics:
                mask = actuals != 0
                if np.any(mask):
                    mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100
                else:
                    mape = np.nan
                series_metrics_dict["mape"] = mape
                all_metrics["mape"].append(mape)
            
            if "smape" in metrics:
                smape = calculate_smape(actuals, predictions)
                series_metrics_dict["smape"] = smape
                all_metrics["smape"].append(smape)
            
            # For MASE, we would need training data
            # if "mase" in metrics and series_id in train_data_dict:
            #     train_data = train_data_dict[series_id]
            #     mase = calculate_mase(actuals, predictions, train_data['y'].values)
            #     series_metrics_dict["mase"] = mase
            #     all_metrics["mase"].append(mase)

            # Store metrics for this series
            series_metrics[series_id] = series_metrics_dict

            # Log metrics for this series
            metric_str = ", ".join([f"{k.upper()}={v:.2f}" + ("%" if k == "mape" or k == "smape" else "") 
                                   for k, v in series_metrics_dict.items() if not np.isnan(v)])
            logger.info(f"Metrics for {series_id}: {metric_str}")

            # Plot the forecast vs actual for this series
            plt.subplot(len(series_ids), 1, i + 1)
            plt.plot(merged_data["ds"], merged_data["y"], "b.", label="Actual")
            plt.plot(
                merged_data["ds"], merged_data["yhat"], "r-", label="Forecast"
            )
            plt.fill_between(
                merged_data["ds"],
                merged_data["yhat_lower"],
                merged_data["yhat_upper"],
                color="gray",
                alpha=0.2,
            )
            plt.title(f"Forecast vs Actual for {series_id}")
            plt.legend()

    # Calculate average metrics across all series
    average_metrics = {}
    
    for metric in metrics:
        if metric in all_metrics and all_metrics[metric]:
            # Filter out NaN values before calculating mean
            valid_values = [v for v in all_metrics[metric] if not np.isnan(v)]
            if valid_values:
                average_metrics[f"avg_{metric}"] = np.mean(valid_values)
                logger.info(f"Final Average {metric.upper()}: {average_metrics[f'avg_{metric}']:.2f}")
            else:
                average_metrics[f"avg_{metric}"] = np.nan
                logger.warning(f"No valid {metric.upper()} values calculated")
        else:
            average_metrics[f"avg_{metric}"] = np.nan
            logger.warning(f"No {metric.upper()} values found")
    
    if not average_metrics:
        logger.error("No valid metrics calculated!")
        average_metrics = {"avg_mae": np.nan, "avg_rmse": np.nan, "avg_mape": np.nan}

    # Save plot to buffer
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Log metrics to ZenML
    log_metadata(
        metadata={
            "avg_mae": float(average_metrics["avg_mae"])
            if not np.isnan(average_metrics["avg_mae"])
            else 0.0,
            "avg_rmse": float(average_metrics["avg_rmse"])
            if not np.isnan(average_metrics["avg_rmse"])
            else 0.0,
            "avg_mape": float(average_metrics["avg_mape"])
            if not np.isnan(average_metrics["avg_mape"])
            else 0.0,
        }
    )

    logger.info(f"Final Average MAE: {average_metrics['avg_mae']:.2f}")
    logger.info(f"Final Average RMSE: {average_metrics['avg_rmse']:.2f}")
    logger.info(
        f"Final Average MAPE: {average_metrics['avg_mape']:.2f}%"
        if not np.isnan(average_metrics["avg_mape"])
        else "Final Average MAPE: N/A"
    )

    # Create HTML report
    html_report = create_evaluation_report(
        average_metrics, series_metrics, plot_data
    )

    return average_metrics, html_report


def create_evaluation_report(average_metrics, series_metrics, plot_image_data):
    """Create an HTML report for model evaluation."""
    # Create a table for series-specific metrics
    series_rows = ""
    for series_id, metrics in series_metrics.items():
        mape_value = (
            f"{metrics['mape']:.2f}%"
            if not np.isnan(metrics.get("mape", np.nan))
            else "N/A"
        )
        series_rows += f"""
        <tr class="border-b">
            <td class="py-2 px-4">{series_id}</td>
            <td class="py-2 px-4 text-right">{metrics["mae"]:.2f}</td>
            <td class="py-2 px-4 text-right">{metrics["rmse"]:.2f}</td>
            <td class="py-2 px-4 text-right">{mape_value}</td>
        </tr>
        """

    # Create overall metrics section
    avg_mape = (
        f"{average_metrics['avg_mape']:.2f}%"
        if not np.isnan(average_metrics.get("avg_mape", np.nan))
        else "N/A"
    )

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prophet Model Evaluation</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .metrics-container {{ display: flex; gap: 20px; margin-bottom: 30px; }}
            .metric-card {{ background: #f8f9fa; border-radius: 8px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); flex: 1; }}
            .metric-value {{ font-size: 28px; font-weight: bold; color: #3498db; margin: 10px 0; }}
            .metric-label {{ font-size: 14px; color: #7f8c8d; text-transform: uppercase; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th {{ background: #eef1f5; text-align: left; padding: 12px; }}
            td {{ padding: 10px; }}
            .series-table {{ max-height: 400px; overflow-y: auto; }}
            .forecast-plot {{ margin-top: 30px; text-align: center; }}
            .forecast-plot img {{ max-width: 100%; height: auto; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <h1>Prophet Model Evaluation Results</h1>
        
        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-label">Average MAE</div>
                <div class="metric-value">{average_metrics["avg_mae"]:.2f}</div>
                <div>Mean Absolute Error</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Average RMSE</div>
                <div class="metric-value">{average_metrics["avg_rmse"]:.2f}</div>
                <div>Root Mean Square Error</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Average MAPE</div>
                <div class="metric-value">{avg_mape}</div>
                <div>Mean Absolute Percentage Error</div>
            </div>
        </div>
        
        <h2>Series-Specific Metrics</h2>
        <div class="series-table">
            <table>
                <thead>
                    <tr>
                        <th>Series ID</th>
                        <th>MAE</th>
                        <th>RMSE</th>
                        <th>MAPE</th>
                    </tr>
                </thead>
                <tbody>
                    {series_rows}
                </tbody>
            </table>
        </div>
        
        <div class="forecast-plot">
            <h2>Forecast Visualization</h2>
            <img src="data:image/png;base64,{plot_image_data}" alt="Forecast Plot">
        </div>
    </body>
    </html>
    """

    return HTMLString(html)
