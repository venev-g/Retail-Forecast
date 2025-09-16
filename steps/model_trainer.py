import itertools
import logging
from typing import Dict, List, Optional

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
    cv_horizon: str = '7 days',
    cv_initial: str = '21 days',
    cv_period: str = '3 days',
    hyperparameter_grid: Optional[Dict] = None,
    custom_seasonalities: Optional[Dict] = None,
    regressors: Optional[Dict] = None,
    cv_settings: Optional[Dict] = None,
    random_seed: int = 42,
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
    
    # Use configurable parameter grid or fallback to defaults
    if hyperparameter_grid:
        param_grid = hyperparameter_grid
        logger.info("Using hyperparameter grid from configuration")
    else:
        # Default parameter grid based on Prophet best practices
        param_grid = {
            'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
            'seasonality_prior_scale': [0.01, 0.1, 1.0, 5.0, 10.0],
            'holidays_prior_scale': [0.1, 1.0, 5.0, 10.0, 20.0],
            'seasonality_mode': ['additive', 'multiplicative'],
        }
        logger.info("Using default hyperparameter grid")
    
    # Generate all parameter combinations
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    
    # Limit number of evaluations to prevent overfitting and save time
    if len(all_params) > max_evals:
        # Select diverse combinations using configurable random seed
        np.random.seed(random_seed)
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
            
            # Add regressors if available using configurable regressor lists
            if regressors:
                regressor_columns = regressors.get('primary', []) + regressors.get('advanced', [])
            else:
                regressor_columns = ['is_weekend', 'is_promo', 'is_month_end', 'is_month_start']
            
            for regressor in regressor_columns:
                if regressor in train_data.columns:
                    m.add_regressor(regressor, mode=params.get('seasonality_mode', 'multiplicative'))
            
            # Add custom seasonalities if configured
            if custom_seasonalities:
                for season_name, season_config in custom_seasonalities.items():
                    if season_config.get('enabled', True):
                        min_points = season_config.get('min_data_points', 28)
                        if len(train_data) >= min_points:
                            m.add_seasonality(
                                name=season_name,
                                period=season_config.get('period', 30.5),
                                fourier_order=season_config.get('fourier_order', 3)
                            )
            else:
                # Default monthly seasonality
                if len(train_data) >= 28:
                    m.add_seasonality(name='monthly', period=30.5, fourier_order=3)
            
            # Fit model
            m.fit(train_data)
            
            # Perform cross-validation with configurable settings
            try:
                # Use configurable CV settings or defaults
                min_cv_data = cv_settings.get('min_data_for_cv', 40) if cv_settings else 40
                split_ratio = cv_settings.get('train_test_split_ratio', 0.8) if cv_settings else 0.8
                
                if len(train_data) >= min_cv_data:  # Sufficient data for CV
                    df_cv = cross_validation(m, initial=cv_initial, period=cv_period, horizon=cv_horizon)
                    df_p = performance_metrics(df_cv)
                    rmse = df_p['rmse'].mean()
                else:
                    # Use simple train-test split for limited data
                    cutoff = int(len(train_data) * split_ratio)
                    train_subset = train_data.iloc[:cutoff]
                    test_subset = train_data.iloc[cutoff:]
                    
                    m_temp = Prophet(**params)
                    for regressor in regressor_columns:
                        if regressor in train_data.columns:
                            m_temp.add_regressor(regressor, mode=params.get('seasonality_mode', 'multiplicative'))
                    
                    # Add custom seasonalities to temp model
                    if custom_seasonalities:
                        for season_name, season_config in custom_seasonalities.items():
                            if season_config.get('enabled', True):
                                min_points = season_config.get('min_data_points')
                                if len(train_data) >= min_points:
                                    m_temp.add_seasonality(
                                        name=season_name,
                                        period=season_config.get('period'),
                                        fourier_order=season_config.get('fourier_order')
                                    )
                    # else:
                    #     # Default seasonality
                    #     if len(train_data) >= 28:
                    #         m_temp.add_seasonality(name='monthly', period=30.5, fourier_order=3)
                    
                    m_temp.fit(train_subset)
                    forecast = m_temp.predict(test_subset)
                    rmse = np.sqrt(np.mean((forecast['yhat'] - test_subset['y']) ** 2))
                
                rmses.append(rmse)
                
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = params
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
    use_regressors: bool = True,
    custom_seasonalities: Optional[Dict] = None,
    regressors: Optional[Dict] = None,
    weekly_seasonality: bool = True,
    yearly_seasonality: bool = False,
    daily_seasonality: bool = False,
    growth: str = "linear",
    interval_width: float = 0.95,
    mcmc_samples: int = 0,
) -> Prophet:
    """
    Create an advanced Prophet model with optimized parameters and features.
    
    Args:
        holiday_dataframe: Holiday dataframe for Prophet
        train_data: Training data
        best_params: Optimized hyperparameters
        series_id: Series identifier for logging
        use_regressors: Whether to add regressors
    
    Returns:
        Configured Prophet model
    """
    # Initialize Prophet with optimized parameters
    # Ensure growth is always linear to avoid cap_scaled issues
    prophet_params = best_params.copy()
    prophet_params.pop('growth', None)  # Remove growth from best_params if present
    
    model = Prophet(
        holidays=holiday_dataframe if not holiday_dataframe.empty else None,
        weekly_seasonality=weekly_seasonality,
        yearly_seasonality=yearly_seasonality,
        daily_seasonality=daily_seasonality,
        growth=growth,
        interval_width=interval_width,
        mcmc_samples=mcmc_samples,
        **prophet_params
    )
    
    # Add custom seasonalities using configuration
    if custom_seasonalities:
        for season_name, season_config in custom_seasonalities.items():
            if season_config.get('enabled', True):
                min_points = season_config.get('min_data_points', 28)
                if len(train_data) >= min_points:
                    model.add_seasonality(
                        name=season_name,
                        period=season_config.get('period', 30.5),
                        fourier_order=season_config.get('fourier_order', 3)
                    )
                    logger.info(f"Added {season_name} seasonality (period={season_config.get('period')}) for {series_id}")
    else:
        # Default retail patterns if no configuration
        if len(train_data) >= 28:  # At least 4 weeks
            model.add_seasonality(name='monthly', period=30.5, fourier_order=3)
            logger.info(f"Added monthly seasonality for {series_id}")
        
        if len(train_data) >= 14:  # At least 2 weeks
            model.add_seasonality(name='biweekly', period=14, fourier_order=2)
            logger.info(f"Added bi-weekly seasonality for {series_id}")
    
    # Add regressors using configuration
    if use_regressors:
        regressor_mode = best_params.get('seasonality_mode', 'multiplicative')
        
        if regressors:
            # Use configured regressors
            primary_regressors = regressors.get('primary', [])
            advanced_regressors = regressors.get('advanced', [])
            special_regressors = regressors.get('special', [])
            
            # Add primary regressors
            for regressor in primary_regressors:
                if regressor in train_data.columns:
                    model.add_regressor(regressor, mode=regressor_mode)
                    logger.info(f"Added {regressor} regressor (mode: {regressor_mode}) for {series_id}")
            
            # Add advanced regressors
            for regressor in advanced_regressors:
                if regressor in train_data.columns:
                    model.add_regressor(regressor, mode=regressor_mode)
                    logger.info(f"Added {regressor} regressor (mode: {regressor_mode}) for {series_id}")
            
            # Add special regressors (with specific modes)
            for regressor in special_regressors:
                if regressor in train_data.columns:
                    # Special regressors like price_change use multiplicative mode
                    special_mode = 'multiplicative' if regressor == 'price_change' else regressor_mode
                    model.add_regressor(regressor, mode=special_mode)
                    logger.info(f"Added {regressor} regressor (mode: {special_mode}) for {series_id}")
        else:
            # Default regressors if no configuration
            default_regressors = ['is_weekend', 'is_promo', 'is_month_end', 'is_month_start']
            for regressor in default_regressors:
                if regressor in train_data.columns:
                    model.add_regressor(regressor, mode=regressor_mode)
                    logger.info(f"Added {regressor} regressor (mode: {regressor_mode}) for {series_id}")
            
            # Default advanced regressors
            advanced_regressors = ['is_payday', 'is_quarter_end', 'is_summer', 'is_winter']
            for regressor in advanced_regressors:
                if regressor in train_data.columns:
                    model.add_regressor(regressor, mode=regressor_mode)
                    logger.info(f"Added {regressor} regressor (mode: {regressor_mode}) for {series_id}")
            
            # Default special regressors
            if 'price_change' in train_data.columns:
                model.add_regressor('price_change', mode='multiplicative')
                logger.info(f"Added price_change regressor for {series_id}")
    
    return model


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
    enable_hyperparameter_tuning: bool = False,
    max_optimization_evals: int = 12,
    use_warm_start: bool = True,
    # New configurable parameters from YAML
    hyperparameter_grid: Optional[Dict] = None,
    custom_seasonalities: Optional[Dict] = None,
    regressors: Optional[Dict] = None,
    cv_settings: Optional[Dict] = None,
    random_seed: int = 42,
    min_training_points: int = 10,
) -> Annotated[Dict[str, Prophet], "trained_prophet_models"]:
    """Train optimized Prophet models for each store-item combination with advanced features.

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
        enable_hyperparameter_tuning: Whether to optimize hyperparameters
        max_optimization_evals: Maximum hyperparameter combinations to test
        use_warm_start: Whether to reuse optimal params across series

    Returns:
        Dictionary of trained Prophet models for each series
    """
    models = {}
    failed_series = []
    global_best_params = None
    
    logger.info(f"Training optimized models for {len(series_ids)} series")
    
    # Use manual parameters if hyperparameter tuning is disabled
    if not enable_hyperparameter_tuning:
        global_best_params = {
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale,
            'holidays_prior_scale': holidays_prior_scale,
            'seasonality_mode': seasonality_mode
        }

    for i, series_id in enumerate(series_ids):
        logger.info(f"Processing series {i+1}/{len(series_ids)}: {series_id}")
        
        try:
            train_data = train_data_dict[series_id].copy()
            
            # Data validation using configurable threshold
            if len(train_data) < min_training_points:
                logger.warning(f"Insufficient training data for {series_id}: {len(train_data)} points (minimum: {min_training_points})")
                failed_series.append(series_id)
                continue
            
            # Always use linear growth - skip capacity/floor setup for logistic growth
            # This avoids the cap_scaled error in Prophet
            
            # Set floor for forecasts only if needed (typically not for linear growth)
            if floor is not None and floor > 0:
                train_data['floor'] = floor

            # Hyperparameter optimization (first series or when not using warm start)
            if enable_hyperparameter_tuning and (global_best_params is None or not use_warm_start):
                logger.info(f"Optimizing hyperparameters for {series_id}")
                
                # Get CV settings for optimization
                cv_horizon = cv_settings.get('horizon', '7 days') if cv_settings else '7 days'
                cv_initial = cv_settings.get('initial', '21 days') if cv_settings else '21 days'
                cv_period = cv_settings.get('period', '3 days') if cv_settings else '3 days'
                
                series_best_params = optimize_hyperparameters(
                    train_data=train_data,
                    holiday_dataframe=holiday_dataframe,
                    max_evals=max_optimization_evals,
                    cv_horizon=cv_horizon,
                    cv_initial=cv_initial,
                    cv_period=cv_period,
                    hyperparameter_grid=hyperparameter_grid,
                    custom_seasonalities=custom_seasonalities,
                    regressors=regressors,
                    cv_settings=cv_settings,
                    random_seed=random_seed,
                )
                
                # Use warm start: reuse best params for subsequent series
                if use_warm_start and global_best_params is None:
                    global_best_params = series_best_params
                    logger.info(f"Using warm start parameters: {global_best_params}")
                elif not use_warm_start:
                    global_best_params = series_best_params
            
            # Use optimized or default parameters
            current_params = global_best_params or {
                'changepoint_prior_scale': changepoint_prior_scale,
                'seasonality_prior_scale': seasonality_prior_scale,
                'holidays_prior_scale': holidays_prior_scale,
                'seasonality_mode': seasonality_mode
            }
            
            logger.info(f"Training {series_id} with parameters: {current_params}")

            # Create advanced Prophet model with configurable parameters
            model = create_advanced_prophet_model(
                holiday_dataframe=holiday_dataframe,
                train_data=train_data,
                best_params=current_params,
                series_id=series_id,
                use_regressors=use_regressors,
                custom_seasonalities=custom_seasonalities,
                regressors=regressors,
                weekly_seasonality=weekly_seasonality,
                yearly_seasonality=yearly_seasonality,
                daily_seasonality=daily_seasonality,
                growth=growth,
                interval_width=interval_width,
                mcmc_samples=mcmc_samples,
            )
            
            # Model is now fully configured through parameters, no hardcoded overrides needed
            
            logger.info(f"Model for {series_id} configured with {len(holiday_dataframe)} holidays and regressors: {use_regressors}")

            # Fit model with error handling
            try:
                model.fit(train_data)
                
                # Validate model fitting
                if hasattr(model, 'params') and model.params is not None:
                    models[series_id] = model
                    logger.info(f"Successfully trained optimized model for {series_id}")
                else:
                    logger.error(f"Model fitting failed for {series_id}: Invalid parameters")
                    failed_series.append(series_id)
                    
            except Exception as fit_error:
                logger.error(f"Model fitting error for {series_id}: {str(fit_error)}")
                failed_series.append(series_id)
                
        except Exception as e:
            logger.error(f"Error processing series {series_id}: {str(e)}")
            failed_series.append(series_id)

    # Results summary
    if failed_series:
        logger.warning(f"Failed to train models for {len(failed_series)} series: {failed_series}")
    
    logger.info(f"Successfully trained {len(models)} optimized Prophet models out of {len(series_ids)} series")
    
    if not models:
        raise ValueError("No models were successfully trained!")
    
    # Log final parameters used
    if global_best_params:
        logger.info(f"Final optimized parameters: {global_best_params}")
    
    return models
