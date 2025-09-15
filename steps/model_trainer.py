import logging
from typing import Dict, List, Optional

import pandas as pd
from materializers.prophet_materializer import ProphetMaterializer
from prophet import Prophet
from typing_extensions import Annotated
from zenml import step

logger = logging.getLogger(__name__)


@step(output_materializers=ProphetMaterializer)
def train_model(
    train_data_dict: Dict[str, pd.DataFrame],
    series_ids: List[str],
    holiday_dataframe: pd.DataFrame,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = False,
    daily_seasonality: bool = False,
    seasonality_mode: str = "multiplicative",
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 1.0,
    holidays_prior_scale: float = 10.0,
    growth: str = "linear",
    interval_width: float = 0.95,
    mcmc_samples: int = 0,
    cap: Optional[float] = None,
    floor: float = 0,
    use_regressors: bool = True,
) -> Annotated[Dict[str, Prophet], "trained_prophet_models"]:
    """Train enhanced Prophet models for each store-item combination.

    Args:
        train_data_dict: Dictionary with training data for each series
        series_ids: List of series identifiers
        holiday_dataframe: DataFrame with holiday information for Prophet
        weekly_seasonality: Whether to include weekly seasonality
        yearly_seasonality: Whether to include yearly seasonality
        daily_seasonality: Whether to include daily seasonality
        seasonality_mode: 'additive' or 'multiplicative'
        changepoint_prior_scale: Flexibility of automatic changepoint selection
        seasonality_prior_scale: Strength of seasonality model
        holidays_prior_scale: Strength of holiday effects
        growth: 'linear' or 'logistic' growth
        interval_width: Width of uncertainty intervals
        mcmc_samples: Number of MCMC samples (0 for MAP estimation)
        cap: Carrying capacity for logistic growth
        floor: Minimum forecast value
        use_regressors: Whether to use additional regressors

    Returns:
        Dictionary of trained Prophet models for each series
    """
    models = {}
    failed_series = []

    for series_id in series_ids:
        logger.info(f"Training model for {series_id}...")
        
        try:
            train_data = train_data_dict[series_id].copy()
            
            # Data validation
            if len(train_data) < 10:
                logger.warning(f"Insufficient training data for {series_id}: {len(train_data)} points")
                failed_series.append(series_id)
                continue
            
            # Set capacity for logistic growth
            if growth == "logistic":
                if cap is None:
                    # Auto-detect capacity based on maximum value + buffer
                    series_cap = train_data['y'].max() * 1.2
                else:
                    series_cap = cap
                train_data['cap'] = series_cap
            
            # Set floor for forecasts
            if floor is not None:
                train_data['floor'] = floor

            # Initialize Prophet model with enhanced parameters including holidays
            model = Prophet(
                holidays=holiday_dataframe if not holiday_dataframe.empty else None,
                weekly_seasonality=weekly_seasonality,
                yearly_seasonality=yearly_seasonality,
                daily_seasonality=daily_seasonality,
                seasonality_mode=seasonality_mode,
                changepoint_prior_scale=changepoint_prior_scale,
                seasonality_prior_scale=seasonality_prior_scale,
                holidays_prior_scale=holidays_prior_scale,
                growth=growth,
                interval_width=interval_width,
                mcmc_samples=mcmc_samples,
            )
            
            # Add custom seasonalities for retail patterns
            if len(train_data) >= 28:  # At least 4 weeks of data
                model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
            
            # Add regressors based on available columns and user preference
            if use_regressors:
                # Main regressors from calendar data
                if 'is_weekend' in train_data.columns:
                    model.add_regressor('is_weekend', mode=seasonality_mode)
                    logger.info(f"Added is_weekend regressor for {series_id}")
                
                if 'is_promo' in train_data.columns:
                    model.add_regressor('is_promo', mode=seasonality_mode)
                    logger.info(f"Added is_promo regressor for {series_id}")
                
                # Additional retail-specific regressors
                if 'is_month_end' in train_data.columns:
                    model.add_regressor('is_month_end', mode=seasonality_mode)
                    logger.info(f"Added is_month_end regressor for {series_id}")
                
                if 'is_month_start' in train_data.columns:
                    model.add_regressor('is_month_start', mode=seasonality_mode)
                    logger.info(f"Added is_month_start regressor for {series_id}")
            
            logger.info(f"Model for {series_id} configured with {len(holiday_dataframe)} holidays and regressors: {use_regressors}")

            # Fit model with error handling
            try:
                model.fit(train_data)
                
                # Validate model fitting
                if hasattr(model, 'params') and model.params is not None:
                    models[series_id] = model
                    logger.info(f"Successfully trained model for {series_id}")
                else:
                    logger.error(f"Model fitting failed for {series_id}: Invalid parameters")
                    failed_series.append(series_id)
                    
            except Exception as fit_error:
                logger.error(f"Model fitting error for {series_id}: {str(fit_error)}")
                failed_series.append(series_id)
                
        except Exception as e:
            logger.error(f"Error processing series {series_id}: {str(e)}")
            failed_series.append(series_id)

    # Log results
    if failed_series:
        logger.warning(f"Failed to train models for {len(failed_series)} series: {failed_series}")
    
    logger.info(f"Successfully trained {len(models)} Prophet models out of {len(series_ids)} series")
    
    if not models:
        raise ValueError("No models were successfully trained!")

    return models
