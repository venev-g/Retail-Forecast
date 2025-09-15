import itertools
import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from materializers.prophet_materializer import ProphetMaterializer
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from typing_extensions import Annotated
from zenml import step

logger = logging.getLogger(__name__)


def optimize_hyperparameters(
    train_data: pd.DataFrame,
    holiday_dataframe: pd.DataFrame,
    max_evals: int = 12,
    cv_horizon: str = '14 days',
    cv_initial: str = '28 days',
    cv_period: str = '7 days',
) -> Dict:
    """
    Optimize Prophet hyperparameters using cross-validation.
    
    Args:
        train_data: Training data in Prophet format
        holiday_dataframe: Holiday dataframe for Prophet
        max_evals: Maximum number of parameter combinations to evaluate
        cv_horizon: Cross-validation forecast horizon
        cv_initial: Initial training period for CV
        cv_period: Period between CV cutoffs
    
    Returns:
        Dictionary with best hyperparameters
    """
    logger.info("Starting hyperparameter optimization...")
    
    # Define parameter grid based on latest Prophet best practices
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 5.0, 10.0],
        'holidays_prior_scale': [0.1, 1.0, 5.0, 10.0, 20.0],
        'seasonality_mode': ['additive', 'multiplicative'],
    }
    
    # Generate all parameter combinations
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    # Limit number of evaluations to prevent overfitting and save time
    if len(all_params) > max_evals:
        # Select diverse combinations
        np.random.seed(42)
        selected_indices = np.random.choice(len(all_params), max_evals, replace=False)
        all_params = [all_params[i] for i in selected_indices]
    
    logger.info(f"Evaluating {len(all_params)} parameter combinations")
    
    best_params = None
    best_rmse = float('inf')
    rmses = []
    
    for i, params in enumerate(all_params):
        try:
            logger.info(f"Testing parameters {i+1}/{len(all_params)}: {params}")
            
            # Initialize Prophet with current parameters
            m = Prophet(
                holidays=holiday_dataframe if not holiday_dataframe.empty else None,
                weekly_seasonality=True,
                yearly_seasonality=False,  # Limited data
                daily_seasonality=False,
                **params
            )
            
            # Add regressors if available
            regressor_columns = ['is_weekend', 'is_promo', 'is_month_end', 'is_month_start']
            for regressor in regressor_columns:
                if regressor in train_data.columns:
                    m.add_regressor(regressor, mode=params.get('seasonality_mode', 'multiplicative'))
            
            # Add monthly seasonality
            if len(train_data) >= 28:
                m.add_seasonality(name='monthly', period=30.5, fourier_order=3)
            
            # Fit model
            m.fit(train_data)
            
            # Perform cross-validation with limited data considerations
            try:
                df_cv = cross_validation(
                    m, 
                    initial=cv_initial,
                    period=cv_period,
                    horizon=cv_horizon,
                    parallel=None  # Avoid parallel issues
                )
                
                # Calculate performance metrics
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmse = df_p['rmse'].mean()  # Use mean RMSE across horizons
                rmses.append(rmse)
                
                # Update best parameters
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params.copy()
                    logger.info(f"New best RMSE: {rmse:.4f} with params: {params}")
                
            except Exception as cv_error:
                logger.warning(f"Cross-validation failed for params {params}: {cv_error}")
                rmses.append(float('inf'))
                
        except Exception as e:
            logger.warning(f"Failed to evaluate params {params}: {e}")
            rmses.append(float('inf'))
    
    # Fallback to default parameters if optimization failed
    if best_params is None:
        logger.warning("Hyperparameter optimization failed, using default parameters")
        best_params = {
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 1.0,
            'holidays_prior_scale': 10.0,
            'seasonality_mode': 'multiplicative'
        }
    else:
        logger.info(f"Best hyperparameters found: {best_params} with RMSE: {best_rmse:.4f}")
    
    return best_params


def create_advanced_prophet_model(
    holiday_dataframe: pd.DataFrame,
    train_data: pd.DataFrame,
    best_params: Dict,
    series_id: str,
) -> Prophet:
    """
    Create an advanced Prophet model with optimized parameters and features.
    
    Args:
        holiday_dataframe: Holiday dataframe for Prophet
        train_data: Training data
        best_params: Optimized hyperparameters
        series_id: Series identifier for logging
    
    Returns:
        Configured Prophet model
    """
    # Initialize Prophet with optimized parameters
    model = Prophet(
        holidays=holiday_dataframe if not holiday_dataframe.empty else None,
        weekly_seasonality=True,
        yearly_seasonality=False,  # Limited to 90 days of data
        daily_seasonality=False,
        growth='linear',
        interval_width=0.95,
        mcmc_samples=0,  # Use MAP estimation for speed
        **best_params
    )
    
    # Add custom seasonalities based on retail patterns
    if len(train_data) >= 28:  # At least 4 weeks
        # Monthly seasonality for end-of-month effects
        model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
        logger.info(f"Added monthly seasonality for {series_id}")
    
    if len(train_data) >= 14:  # At least 2 weeks
        # Bi-weekly patterns (payday effects)
        model.add_seasonality(name='biweekly', period=14, fourier_order=2)
        logger.info(f"Added bi-weekly seasonality for {series_id}")
    
    # Add regressors with mode matching seasonality_mode
    regressor_mode = best_params.get('seasonality_mode', 'multiplicative')
    
    # Primary regressors
    if 'is_weekend' in train_data.columns:
        model.add_regressor('is_weekend', mode=regressor_mode)
        logger.info(f"Added is_weekend regressor (mode: {regressor_mode}) for {series_id}")
    
    if 'is_promo' in train_data.columns:
        model.add_regressor('is_promo', mode=regressor_mode)
        logger.info(f"Added is_promo regressor (mode: {regressor_mode}) for {series_id}")
    
    # Secondary regressors for retail patterns
    if 'is_month_end' in train_data.columns:
        model.add_regressor('is_month_end', mode=regressor_mode)
        logger.info(f"Added is_month_end regressor (mode: {regressor_mode}) for {series_id}")
    
    if 'is_month_start' in train_data.columns:
        model.add_regressor('is_month_start', mode=regressor_mode)
        logger.info(f"Added is_month_start regressor (mode: {regressor_mode}) for {series_id}")
    
    # Add price elasticity if price data is available
    if 'price_change' in train_data.columns:
        model.add_regressor('price_change', mode='multiplicative')
        logger.info(f"Added price_change regressor for {series_id}")
    
    return model


@step(output_materializers=ProphetMaterializer)
def train_optimized_model(
    train_data_dict: Dict[str, pd.DataFrame],
    series_ids: List[str],
    holiday_dataframe: pd.DataFrame,
    enable_hyperparameter_tuning: bool = True,
    max_optimization_evals: int = 12,
    use_warm_start: bool = True,
) -> Annotated[Dict[str, Prophet], "trained_prophet_models"]:
    """
    Train optimized Prophet models with hyperparameter tuning and advanced features.
    
    Args:
        train_data_dict: Dictionary with training data for each series
        series_ids: List of series identifiers
        holiday_dataframe: DataFrame with holiday information
        enable_hyperparameter_tuning: Whether to optimize hyperparameters
        max_optimization_evals: Maximum hyperparameter combinations to test
        use_warm_start: Whether to use warm-start optimization
    
    Returns:
        Dictionary of trained Prophet models
    """
    models = {}
    failed_series = []
    global_best_params = None
    
    logger.info(f"Training optimized models for {len(series_ids)} series")
    
    for i, series_id in enumerate(series_ids):
        logger.info(f"Processing series {i+1}/{len(series_ids)}: {series_id}")
        
        try:
            train_data = train_data_dict[series_id].copy()
            
            # Data validation
            if len(train_data) < 21:
                logger.warning(f"Insufficient training data for {series_id}: {len(train_data)} points")
                failed_series.append(series_id)
                continue
            
            # Hyperparameter optimization (only for first series or if not using warm start)
            if enable_hyperparameter_tuning and (global_best_params is None or not use_warm_start):
                try:
                    logger.info(f"Optimizing hyperparameters for {series_id}")
                    best_params = optimize_hyperparameters(
                        train_data=train_data,
                        holiday_dataframe=holiday_dataframe,
                        max_evals=max_optimization_evals,
                        cv_horizon='7 days',  # Shorter horizon for limited data
                        cv_initial='21 days',  # Minimum initial period
                        cv_period='3 days'    # More frequent evaluation
                    )
                    
                    # Store globally if first optimization
                    if global_best_params is None:
                        global_best_params = best_params
                        logger.info(f"Set global best parameters: {global_best_params}")
                        
                except Exception as opt_error:
                    logger.warning(f"Hyperparameter optimization failed for {series_id}: {opt_error}")
                    best_params = {
                        'changepoint_prior_scale': 0.05,
                        'seasonality_prior_scale': 1.0,
                        'holidays_prior_scale': 10.0,
                        'seasonality_mode': 'multiplicative'
                    }
            else:
                # Use global best parameters or defaults
                best_params = global_best_params or {
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 1.0,
                    'holidays_prior_scale': 10.0,
                    'seasonality_mode': 'multiplicative'
                }
                logger.info(f"Using cached parameters for {series_id}: {best_params}")
            
            # Create advanced Prophet model
            model = create_advanced_prophet_model(
                holiday_dataframe=holiday_dataframe,
                train_data=train_data,
                best_params=best_params,
                series_id=series_id
            )
            
            # Data preprocessing for better fitting
            # Handle potential capacity for logistic growth (future enhancement)
            if 'cap' in train_data.columns or best_params.get('growth') == 'logistic':
                if 'cap' not in train_data.columns:
                    train_data['cap'] = train_data['y'].max() * 1.2
                if 'floor' not in train_data.columns:
                    train_data['floor'] = 0
            
            # Fit the model
            try:
                logger.info(f"Fitting model for {series_id} with {len(train_data)} data points")
                model.fit(train_data)
                
                # Validate model
                if hasattr(model, 'params') and model.params is not None:
                    models[series_id] = model
                    logger.info(f"Successfully trained optimized model for {series_id}")
                else:
                    logger.error(f"Model validation failed for {series_id}")
                    failed_series.append(series_id)
                    
            except Exception as fit_error:
                logger.error(f"Model fitting failed for {series_id}: {fit_error}")
                failed_series.append(series_id)
                
        except Exception as e:
            logger.error(f"Error processing series {series_id}: {e}")
            failed_series.append(series_id)
    
    # Results summary
    if failed_series:
        logger.warning(f"Failed to train models for {len(failed_series)} series: {failed_series}")
    
    logger.info(f"Successfully trained {len(models)} optimized Prophet models")
    
    if not models:
        raise ValueError("No models were successfully trained!")
    
    # Log final parameters used
    if global_best_params:
        logger.info(f"Final optimized parameters: {global_best_params}")
    
    return models