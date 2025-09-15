import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from typing_extensions import Annotated
from zenml import step

logger = logging.getLogger(__name__)


def detect_outliers_advanced(series: pd.Series, method: str = 'iqr', threshold: float = 3.0) -> pd.Series:
    """
    Advanced outlier detection using multiple methods.
    
    Args:
        series: Time series data
        method: Detection method ('iqr', 'zscore', 'modified_zscore')
        threshold: Threshold multiplier
    
    Returns:
        Boolean series indicating outliers
    """
    if method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    elif method == 'modified_zscore':
        median = series.median()
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def load_calendar_features() -> pd.DataFrame:
    """Load and validate calendar features from the calendar.csv file."""
    try:
        data_dir = os.path.join(os.getcwd(), "data")
        calendar_path = os.path.join(data_dir, "calendar.csv")
        
        if os.path.exists(calendar_path):
            calendar_df = pd.read_csv(calendar_path)
            calendar_df['date'] = pd.to_datetime(calendar_df['date'])
            
            # Validate required columns
            required_cols = ['date', 'weekday', 'month', 'is_weekend', 'is_holiday', 'is_promo']
            missing_cols = [col for col in required_cols if col not in calendar_df.columns]
            
            if missing_cols:
                logger.warning(f"Missing calendar columns: {missing_cols}")
                # Generate missing columns
                for col in missing_cols:
                    if col == 'weekday':
                        calendar_df['weekday'] = calendar_df['date'].dt.dayofweek
                    elif col == 'month':
                        calendar_df['month'] = calendar_df['date'].dt.month
                    elif col == 'is_weekend':
                        calendar_df['is_weekend'] = calendar_df['date'].dt.dayofweek.isin([5, 6]).astype(int)
                    elif col == 'is_holiday':
                        calendar_df['is_holiday'] = 0
                    elif col == 'is_promo':
                        calendar_df['is_promo'] = 0
            
            logger.info(f"Loaded calendar features with shape: {calendar_df.shape}")
            logger.info(f"Calendar date range: {calendar_df['date'].min()} to {calendar_df['date'].max()}")
            return calendar_df
        else:
            logger.warning("calendar.csv not found, will generate basic calendar features")
            return None
    except Exception as e:
        logger.error(f"Error loading calendar features: {e}")
        return None


def create_enhanced_holiday_dataframe(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create an enhanced holiday dataframe for Prophet with specific holiday names.
    
    Args:
        calendar_df: Calendar dataframe with is_holiday column
        
    Returns:
        DataFrame with columns 'ds' and 'holiday' for Prophet
    """
    if calendar_df is None or 'is_holiday' not in calendar_df.columns:
        return pd.DataFrame(columns=['ds', 'holiday'])
    
    # Get holiday dates
    holiday_dates = calendar_df[calendar_df['is_holiday'] == 1].copy()
    
    holidays = []
    for _, row in holiday_dates.iterrows():
        date = row['date']
        
        # Enhanced holiday naming based on retail patterns
        holiday_name = 'General Holiday'
        
        # Month-based holiday classification
        if date.month == 1:
            if date.day == 1:
                holiday_name = 'New Year'
            elif date.day <= 15:
                holiday_name = 'January Holiday'
            else:
                holiday_name = 'Late January Holiday'
        elif date.month == 2:
            if date.day <= 14:
                holiday_name = 'February Holiday'
            else:
                holiday_name = 'Late February Holiday'
        elif date.month == 3:
            holiday_name = 'March Holiday'
        elif date.month in [11, 12]:
            holiday_name = 'Year End Holiday'
        elif date.month in [6, 7, 8]:
            holiday_name = 'Summer Holiday'
        else:
            holiday_name = f'Month {date.month} Holiday'
        
        holidays.append({
            'ds': date,
            'holiday': holiday_name
        })
    
    holiday_df = pd.DataFrame(holidays) if holidays else pd.DataFrame(columns=['ds', 'holiday'])
    
    # Add prior and post-holiday effects
    if not holiday_df.empty:
        extended_holidays = []
        for _, row in holiday_df.iterrows():
            # Main holiday
            extended_holidays.append(row.to_dict())
            
            # Pre-holiday effect (day before)
            pre_date = row['ds'] - pd.Timedelta(days=1)
            extended_holidays.append({
                'ds': pre_date,
                'holiday': f"Pre-{row['holiday']}"
            })
            
            # Post-holiday effect (day after)
            post_date = row['ds'] + pd.Timedelta(days=1)
            extended_holidays.append({
                'ds': post_date,
                'holiday': f"Post-{row['holiday']}"
            })
        
        holiday_df = pd.DataFrame(extended_holidays)
    
    logger.info(f"Created enhanced holiday dataframe with {len(holiday_df)} holiday entries")
    return holiday_df


def add_advanced_features(data: pd.DataFrame, calendar_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add advanced time-based and retail-specific features for Prophet.
    
    Args:
        data: Input dataframe with ds and y columns
        calendar_df: Calendar dataframe with additional features
    
    Returns:
        Enhanced dataframe with additional features
    """
    data = data.copy()
    
    # Merge with calendar data if available
    if calendar_df is not None:
        data_with_calendar = data.merge(
            calendar_df[['date', 'weekday', 'month', 'is_weekend', 'is_holiday', 'is_promo']], 
            left_on='ds', 
            right_on='date', 
            how='left'
        ).drop('date', axis=1)
        
        # Fill missing values for dates not in calendar
        data_with_calendar['weekday'] = data_with_calendar['weekday'].fillna(data['ds'].dt.dayofweek)
        data_with_calendar['month'] = data_with_calendar['month'].fillna(data['ds'].dt.month)
        data_with_calendar['is_weekend'] = data_with_calendar['is_weekend'].fillna(
            data['ds'].dt.dayofweek.isin([5, 6]).astype(int)
        )
        data_with_calendar['is_holiday'] = data_with_calendar['is_holiday'].fillna(0)
        data_with_calendar['is_promo'] = data_with_calendar['is_promo'].fillna(0)
        
        data = data_with_calendar
    else:
        # Generate basic features if no calendar data
        data['weekday'] = data['ds'].dt.dayofweek
        data['month'] = data['ds'].dt.month
        data['is_weekend'] = data['weekday'].isin([5, 6]).astype(int)
        data['is_holiday'] = 0
        data['is_promo'] = 0
    
    # Enhanced retail-specific features
    data['is_month_end'] = (data['ds'].dt.day >= 28).astype(int)
    data['is_month_start'] = (data['ds'].dt.day <= 3).astype(int)
    data['is_quarter_end'] = data['ds'].dt.month.isin([3, 6, 9, 12]).astype(int)
    data['is_year_end'] = data['ds'].dt.month.isin([11, 12]).astype(int)
    
    # Week-based features
    data['week_of_month'] = (data['ds'].dt.day - 1) // 7 + 1
    data['is_first_week'] = (data['week_of_month'] == 1).astype(int)
    data['is_last_week'] = (data['week_of_month'] >= 4).astype(int)
    
    # Payday effects (assuming mid-month and end-of-month paydays)
    data['is_payday'] = ((data['ds'].dt.day >= 14) & (data['ds'].dt.day <= 16) | 
                         (data['ds'].dt.day >= 28)).astype(int)
    
    # Seasonal indicators
    data['is_summer'] = data['ds'].dt.month.isin([6, 7, 8]).astype(int)
    data['is_winter'] = data['ds'].dt.month.isin([12, 1, 2]).astype(int)
    data['is_spring'] = data['ds'].dt.month.isin([3, 4, 5]).astype(int)
    data['is_fall'] = data['ds'].dt.month.isin([9, 10, 11]).astype(int)
    
    # Advanced time features
    data['day_of_year'] = data['ds'].dt.dayofyear
    data['week_of_year'] = data['ds'].dt.isocalendar().week
    
    # Lag features for trend analysis (not used as regressors but helpful for validation)
    data['sales_lag_1'] = data['y'].shift(1)
    data['sales_lag_7'] = data['y'].shift(7)
    
    # Moving averages for trend smoothing
    data['sales_ma_3'] = data['y'].rolling(window=3, min_periods=1).mean()
    data['sales_ma_7'] = data['y'].rolling(window=7, min_periods=1).mean()
    data['sales_ma_14'] = data['y'].rolling(window=14, min_periods=1).mean()
    
    # Price-related features (if price data is available)
    if 'price' in data.columns:
        # Price change indicators
        data['price_change'] = data['price'].pct_change().fillna(0)
        data['price_spike'] = (data['price_change'] > 0.1).astype(int)  # 10% price increase
        data['price_drop'] = (data['price_change'] < -0.1).astype(int)  # 10% price decrease
        
        # Price level indicators
        price_median = data['price'].median()
        data['high_price'] = (data['price'] > price_median * 1.1).astype(int)
        data['low_price'] = (data['price'] < price_median * 0.9).astype(int)
    
    # Ensure all regressor columns are numeric and properly formatted
    regressor_columns = [
        'is_weekend', 'is_holiday', 'is_promo', 'is_month_end', 'is_month_start',
        'is_quarter_end', 'is_year_end', 'is_first_week', 'is_last_week', 'is_payday',
        'is_summer', 'is_winter', 'is_spring', 'is_fall'
    ]
    
    # Add price-related regressors if available
    if 'price' in data.columns:
        regressor_columns.extend(['price_change', 'price_spike', 'price_drop', 'high_price', 'low_price'])
    
    for col in regressor_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
            # Ensure binary features are 0 or 1
            if col.startswith('is_') or col.endswith('_spike') or col.endswith('_drop'):
                data[col] = data[col].astype(int)
    
    logger.info(f"Added advanced features. Total columns: {len(data.columns)}")
    logger.info(f"Regressor columns: {[col for col in regressor_columns if col in data.columns]}")
    
    return data


def apply_data_quality_improvements(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data quality improvements including scaling and normalization.
    
    Args:
        data: Input dataframe
    
    Returns:
        Improved dataframe
    """
    data = data.copy()
    
    # Handle missing values in target variable
    if data['y'].isnull().any():
        logger.warning("Found missing values in target variable, applying interpolation")
        data['y'] = data['y'].interpolate(method='time')
    
    # Ensure positive values (sales cannot be negative)
    if (data['y'] < 0).any():
        logger.warning("Found negative sales values, setting to 0")
        data['y'] = data['y'].clip(lower=0)
    
    # Handle zero variance (constant values)
    if data['y'].std() == 0:
        logger.warning("Zero variance in target variable, adding small random noise")
        noise = np.random.normal(0, data['y'].mean() * 0.001, len(data))
        data['y'] = data['y'] + noise
    
    # Smooth extreme outliers rather than removing them
    y_median = data['y'].median()
    y_mad = np.median(np.abs(data['y'] - y_median))
    
    if y_mad > 0:
        outlier_threshold = 5  # More conservative than before
        outlier_mask = np.abs(data['y'] - y_median) > outlier_threshold * y_mad
        
        if outlier_mask.any():
            logger.info(f"Smoothing {outlier_mask.sum()} extreme outliers")
            # Cap outliers at reasonable bounds
            upper_bound = y_median + outlier_threshold * y_mad
            lower_bound = max(0, y_median - outlier_threshold * y_mad)
            data.loc[outlier_mask, 'y'] = np.clip(data.loc[outlier_mask, 'y'], lower_bound, upper_bound)
    
    return data


@step
def preprocess_data_advanced(
    sales_data: pd.DataFrame,
    test_size: float = 0.2,
    min_train_size: int = 50,
    outlier_method: str = 'modified_zscore',
    outlier_threshold: float = 3.5,
    enable_advanced_features: bool = True,
) -> Tuple[
    Annotated[Dict[str, pd.DataFrame], "training_data"],
    Annotated[Dict[str, pd.DataFrame], "testing_data"],
    Annotated[List[str], "series_identifiers"],
    Annotated[pd.DataFrame, "holiday_dataframe"],
]:
    """
    Advanced data preprocessing for forecasting with Prophet.

    Args:
        sales_data: Raw sales data
        test_size: Proportion of data for testing
        min_train_size: Minimum training points required
        outlier_method: Method for outlier detection
        outlier_threshold: Threshold for outlier detection
        enable_advanced_features: Whether to add advanced features

    Returns:
        Enhanced training and testing data with optimized features
    """
    logger.info(f"Starting advanced preprocessing of sales data with shape: {sales_data.shape}")

    # Load and validate calendar features
    calendar_df = load_calendar_features()
    
    # Create enhanced holiday dataframe
    holiday_df = create_enhanced_holiday_dataframe(calendar_df)

    # Data type conversion and validation
    sales_data["date"] = pd.to_datetime(sales_data["date"])
    sales_data["series_id"] = sales_data["store"] + "-" + sales_data["item"]

    # Get unique series
    series_ids = sales_data["series_id"].unique().tolist()
    logger.info(f"Found {len(series_ids)} unique store-item combinations")

    train_data_dict = {}
    test_data_dict = {}
    processed_series = 0

    for series_id in series_ids:
        try:
            # Filter and prepare series data
            series_data = sales_data[sales_data["series_id"] == series_id].copy()
            series_data = series_data.sort_values("date").drop_duplicates(subset=["date"])

            # Create Prophet format
            prophet_data = series_data[["date", "sales"]].rename(
                columns={"date": "ds", "sales": "y"}
            )
            
            # Add price data if available
            if "price" in series_data.columns:
                prophet_data["price"] = series_data["price"].values

            # Basic data validation
            prophet_data = prophet_data.dropna(subset=['y'])
            prophet_data = prophet_data[prophet_data['y'] >= 0]
            
            if len(prophet_data) < max(21, min_train_size // 2):
                logger.warning(f"Insufficient data for {series_id}: {len(prophet_data)} points")
                continue
            
            # Apply data quality improvements
            prophet_data = apply_data_quality_improvements(prophet_data)
            
            # Advanced outlier detection and handling
            if len(prophet_data) > 10:
                outliers = detect_outliers_advanced(
                    prophet_data['y'], 
                    method=outlier_method, 
                    threshold=outlier_threshold
                )
                
                if outliers.sum() > 0:
                    logger.info(f"Detected {outliers.sum()} outliers in {series_id} using {outlier_method}")
                    # Smooth outliers instead of removing
                    median_val = prophet_data['y'].median()
                    mad_val = np.median(np.abs(prophet_data['y'] - median_val))
                    
                    if mad_val > 0:
                        upper_bound = median_val + outlier_threshold * mad_val
                        lower_bound = max(0, median_val - outlier_threshold * mad_val)
                        prophet_data.loc[outliers, 'y'] = np.clip(
                            prophet_data.loc[outliers, 'y'], 
                            lower_bound, 
                            upper_bound
                        )

            # Add advanced features
            if enable_advanced_features:
                prophet_data = add_advanced_features(prophet_data, calendar_df)

            # Intelligent train/test split considering seasonality
            min_test_points = max(7, int(len(prophet_data) * test_size))  # At least 1 week
            required_train_points = max(min_train_size, len(prophet_data) - min_test_points)
            
            if len(prophet_data) < required_train_points + min_test_points:
                logger.warning(f"Cannot create proper split for {series_id}: {len(prophet_data)} points")
                continue
                
            # Time-based split that preserves recent patterns
            cutoff_idx = len(prophet_data) - min_test_points
            train_data = prophet_data.iloc[:cutoff_idx].copy()
            test_data = prophet_data.iloc[cutoff_idx:].copy()

            # Final validation
            if len(train_data) < min_train_size or len(test_data) == 0:
                logger.warning(f"Invalid split for {series_id}: train={len(train_data)}, test={len(test_data)}")
                continue
            
            # Store processed data
            train_data_dict[series_id] = train_data
            test_data_dict[series_id] = test_data
            processed_series += 1

            if processed_series <= 3:  # Log details for first few series
                logger.info(f"Series {series_id}: train={len(train_data)}, test={len(test_data)}")
                logger.info(f"  Date range: {train_data['ds'].min()} to {test_data['ds'].max()}")
                logger.info(f"  Sales range: {train_data['y'].min():.2f} to {train_data['y'].max():.2f}")

        except Exception as e:
            logger.error(f"Error processing series {series_id}: {e}")
            continue

    if not train_data_dict:
        raise ValueError("No valid series data after advanced preprocessing!")

    logger.info(f"Advanced preprocessing completed: {len(train_data_dict)} series ready for training")
    logger.info(f"Holiday dataframe contains {len(holiday_df)} holiday entries")

    return train_data_dict, test_data_dict, list(train_data_dict.keys()), holiday_df