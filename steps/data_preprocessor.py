import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from typing_extensions import Annotated
from zenml import step

logger = logging.getLogger(__name__)


def detect_outliers_advanced(series: pd.Series, method: str = 'modified_zscore', threshold: float = 3.5) -> pd.Series:
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
        if mad == 0:
            return pd.Series([False] * len(series), index=series.index)
        modified_z_scores = 0.6745 * (series - median) / mad
        return np.abs(modified_z_scores) > threshold
    
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def detect_outliers(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using IQR method (legacy compatibility)."""
    return detect_outliers_advanced(series, method='iqr', threshold=threshold)


def load_calendar_features() -> pd.DataFrame:
    """Load calendar features from the calendar.csv file."""
    try:
        data_dir = os.path.join(os.getcwd(), "data")
        calendar_path = os.path.join(data_dir, "calendar.csv")
        
        if os.path.exists(calendar_path):
            calendar_df = pd.read_csv(calendar_path)
            calendar_df['date'] = pd.to_datetime(calendar_df['date'])
            logger.info(f"Loaded calendar features with shape: {calendar_df.shape}")
            return calendar_df
        else:
            logger.warning("calendar.csv not found, generating basic calendar features")
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
            else:
                holiday_name = 'January Sale Period'
        elif date.month == 2:
            if date.day <= 14:
                holiday_name = 'Valentines Period'
            else:
                holiday_name = 'February Holiday'
        elif date.month == 3:
            holiday_name = 'Spring Holiday'
        elif date.month in [11, 12]:
            holiday_name = 'Holiday Shopping Season'
        elif date.month in [6, 7, 8]:
            holiday_name = 'Summer Holiday'
        else:
            holiday_name = 'Seasonal Holiday'
        
        holidays.append({
            'ds': date,
            'holiday': holiday_name
        })
    
    holiday_df = pd.DataFrame(holidays) if holidays else pd.DataFrame(columns=['ds', 'holiday'])
    
    # Add prior and post-holiday effects
    if not holiday_df.empty:
        extended_holidays = []
        for _, row in holiday_df.iterrows():
            base_date = row['ds']
            base_holiday = row['holiday']
            
            # Add the main holiday
            extended_holidays.append({'ds': base_date, 'holiday': base_holiday})
            
            # Add pre-holiday effect (1 day before)
            pre_date = base_date - pd.Timedelta(days=1)
            extended_holidays.append({'ds': pre_date, 'holiday': f"{base_holiday}_pre"})
            
            # Add post-holiday effect (1 day after)
            post_date = base_date + pd.Timedelta(days=1)
            extended_holidays.append({'ds': post_date, 'holiday': f"{base_holiday}_post"})
        
        holiday_df = pd.DataFrame(extended_holidays)
    
    logger.info(f"Created enhanced holiday dataframe with {len(holiday_df)} holiday entries")
    return holiday_df


def create_holiday_dataframe(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Create a holiday dataframe for Prophet from calendar data (legacy compatibility)."""
    return create_enhanced_holiday_dataframe(calendar_df)


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
    
    # Moving averages for trend analysis
    data['sales_ma_3'] = data['y'].rolling(window=3, min_periods=1).mean()
    data['sales_ma_7'] = data['y'].rolling(window=7, min_periods=1).mean()
    data['sales_ma_14'] = data['y'].rolling(window=14, min_periods=1).mean()
    
    # Price-related features (if price data is available)
    if 'price' in data.columns:
        data['price_change'] = data['price'].pct_change().fillna(0)
        data['price_ma_7'] = data['price'].rolling(window=7, min_periods=1).mean()
        
        # Price elasticity indicator
        if len(data) > 1:
            data['price_elasticity'] = data['y'].rolling(3).corr(data['price']).fillna(0)
        else:
            data['price_elasticity'] = 0
    
    # Ensure all regressor columns are numeric and properly formatted
    regressor_columns = [
        'is_weekend', 'is_holiday', 'is_promo', 'is_month_end', 'is_month_start',
        'is_quarter_end', 'is_year_end', 'is_first_week', 'is_last_week', 'is_payday',
        'is_summer', 'is_winter', 'is_spring', 'is_fall'
    ]
    
    # Add price-related regressors if available
    if 'price' in data.columns:
        regressor_columns.extend(['price_change'])
    
    for col in regressor_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)
            # Ensure boolean columns are integers
            if data[col].dtype == bool or col.startswith('is_'):
                data[col] = data[col].astype(int)
    
    logger.info(f"Added advanced features. Total columns: {len(data.columns)}")
    logger.info(f"Regressor columns: {[col for col in regressor_columns if col in data.columns]}")
    
    return data


def add_features(data: pd.DataFrame, calendar_df: pd.DataFrame = None) -> pd.DataFrame:
    """Add time-based and retail-specific features for Prophet (legacy compatibility)."""
    return add_advanced_features(data, calendar_df)


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
        data['y'] = data['y'].fillna(data['y'].median())
    
    # Ensure positive values (sales cannot be negative)
    if (data['y'] < 0).any():
        data.loc[data['y'] < 0, 'y'] = 0
    
    # Handle zero variance (constant values)
    if data['y'].std() == 0:
        logger.warning("Zero variance in target variable, adding small noise")
        data['y'] += np.random.normal(0, 0.01, len(data))
    
    # Smooth extreme outliers rather than removing them
    y_median = data['y'].median()
    y_mad = np.median(np.abs(data['y'] - y_median))
    
    if y_mad > 0:
        threshold = 3.5
        outliers = np.abs(data['y'] - y_median) > threshold * y_mad
        if outliers.sum() > 0:
            logger.info(f"Smoothing {outliers.sum()} extreme outliers")
            # Cap outliers at reasonable bounds
            upper_bound = y_median + threshold * y_mad
            lower_bound = max(0, y_median - threshold * y_mad)
            data.loc[outliers, 'y'] = np.clip(data.loc[outliers, 'y'], lower_bound, upper_bound)
    
    return data


@step
def preprocess_data(
    sales_data: pd.DataFrame,
    test_size: float = 0.2,
    min_train_size: int = 50,
    outlier_threshold: float = 3.5,
    outlier_method: str = 'modified_zscore',
    enable_feature_engineering: bool = True,
    enable_advanced_features: bool = True,
) -> Tuple[
    Annotated[Dict[str, pd.DataFrame], "training_data"],
    Annotated[Dict[str, pd.DataFrame], "testing_data"],
    Annotated[List[str], "series_identifiers"],
    Annotated[pd.DataFrame, "holiday_dataframe"],
]:
    """Prepare data for forecasting with Prophet with advanced preprocessing.

    Args:
        sales_data: Raw sales data with date, store, item, and sales columns
        test_size: Proportion of data to use for testing
        min_train_size: Minimum number of training points required
        outlier_threshold: Threshold for outlier detection
        outlier_method: Method for outlier detection ('iqr', 'zscore', 'modified_zscore')
        enable_feature_engineering: Whether to add additional features
        enable_advanced_features: Whether to add advanced retail-specific features

    Returns:
        train_data_dict: Dictionary of training dataframes for each series
        test_data_dict: Dictionary of test dataframes for each series
        series_ids: List of unique series identifiers (store-item combinations)
        holiday_dataframe: Holiday dataframe for Prophet
    """
    logger.info(f"Preprocessing sales data with shape: {sales_data.shape}")

    # Load calendar features
    calendar_df = load_calendar_features()
    
    # Create holiday dataframe for Prophet
    holiday_df = create_holiday_dataframe(calendar_df)

    # Convert date to datetime
    sales_data["date"] = pd.to_datetime(sales_data["date"])

    # Create unique series ID for each store-item combination
    sales_data["series_id"] = sales_data["store"] + "-" + sales_data["item"]

    # Get list of unique series
    series_ids = sales_data["series_id"].unique().tolist()
    logger.info(f"Found {len(series_ids)} unique store-item combinations")

    # Create Prophet-formatted dataframes (ds, y) for each series
    train_data_dict = {}
    test_data_dict = {}

    for series_id in series_ids:
        # Filter data for this series
        series_data = sales_data[sales_data["series_id"] == series_id].copy()

        # Sort by date and drop any duplicates
        series_data = series_data.sort_values("date").drop_duplicates(
            subset=["date"]
        )

        # Rename columns for Prophet
        prophet_data = series_data[["date", "sales"]].rename(
            columns={"date": "ds", "sales": "y"}
        )

        # Ensure no NaN values and positive sales
        prophet_data = prophet_data.dropna()
        prophet_data = prophet_data[prophet_data['y'] >= 0]  # Remove negative sales
        
        # Check minimum data requirements
        if len(prophet_data) < max(10, min_train_size // 2):
            logger.warning(f"Insufficient data for series {series_id}: {len(prophet_data)} points, skipping")
            continue
        
        # Advanced outlier detection and handling
        outliers = detect_outliers_advanced(prophet_data['y'], method=outlier_method, threshold=outlier_threshold)
        if outliers.sum() > 0:
            logger.info(f"Detected {outliers.sum()} outliers in {series_id} using {outlier_method} method")
            # Cap outliers instead of removing them to preserve data
            if outlier_method == 'iqr':
                Q1 = prophet_data['y'].quantile(0.25)
                Q3 = prophet_data['y'].quantile(0.75)
                IQR = Q3 - Q1
                upper_bound = Q3 + outlier_threshold * IQR
                lower_bound = max(0, Q1 - outlier_threshold * IQR)
            else:
                # For z-score methods, use median and MAD
                median = prophet_data['y'].median()
                mad = np.median(np.abs(prophet_data['y'] - median))
                upper_bound = median + outlier_threshold * mad
                lower_bound = max(0, median - outlier_threshold * mad)
            
            prophet_data.loc[outliers, 'y'] = np.clip(prophet_data.loc[outliers, 'y'], lower_bound, upper_bound)
        
        # Apply data quality improvements
        prophet_data = apply_data_quality_improvements(prophet_data)
        
        # Add features if enabled
        if enable_feature_engineering:
            if enable_advanced_features:
                prophet_data = add_advanced_features(prophet_data, calendar_df)
            else:
                prophet_data = add_features(prophet_data, calendar_df)

        # Ensure we have enough training data
        min_test_points = max(1, int(len(prophet_data) * test_size))
        required_train_points = max(min_train_size, len(prophet_data) - min_test_points)
        
        if len(prophet_data) < required_train_points + min_test_points:
            logger.warning(f"Insufficient data for proper train/test split in {series_id}: {len(prophet_data)} points")
            continue
            
        # Use time-based split that respects seasonality
        cutoff_idx = len(prophet_data) - min_test_points
        
        # Split into train and test
        train_data = prophet_data.iloc[:cutoff_idx].copy()
        test_data = prophet_data.iloc[cutoff_idx:].copy()

        # Final validation
        if len(train_data) < min_train_size or len(test_data) == 0:
            logger.warning(f"Invalid split for series {series_id}: train={len(train_data)}, test={len(test_data)}")
            continue
        
        # Data quality checks
        if train_data['y'].std() == 0:
            logger.warning(f"Zero variance in training data for {series_id}, adding small noise")
            train_data['y'] += np.random.normal(0, 0.01, len(train_data))

        # Store in dictionaries
        train_data_dict[series_id] = train_data
        test_data_dict[series_id] = test_data

        logger.info(
            f"Series {series_id}: {len(train_data)} train points, {len(test_data)} test points"
        )

    if not train_data_dict:
        raise ValueError("No valid series data after preprocessing!")

    # Get a sample series to print details
    sample_id = next(iter(train_data_dict))
    sample_train = train_data_dict[sample_id]
    sample_test = test_data_dict[sample_id]

    logger.info(f"Sample series {sample_id}:")
    logger.info(f"  Train data shape: {sample_train.shape}")
    logger.info(
        f"  Train date range: {sample_train['ds'].min()} to {sample_train['ds'].max()}"
    )
    logger.info(f"  Test data shape: {sample_test.shape}")
    logger.info(
        f"  Test date range: {sample_test['ds'].min()} to {sample_test['ds'].max()}"
    )

    return train_data_dict, test_data_dict, series_ids, holiday_df
