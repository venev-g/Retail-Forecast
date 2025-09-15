import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from typing_extensions import Annotated
from zenml import step

logger = logging.getLogger(__name__)


def detect_outliers(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Detect outliers using IQR method."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    return (series < lower_bound) | (series > upper_bound)


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


def create_holiday_dataframe(calendar_df: pd.DataFrame) -> pd.DataFrame:
    """Create a holiday dataframe for Prophet from calendar data.
    
    Args:
        calendar_df: Calendar dataframe with is_holiday column
        
    Returns:
        DataFrame with columns 'ds' and 'holiday' for Prophet
    """
    if calendar_df is None or 'is_holiday' not in calendar_df.columns:
        return pd.DataFrame(columns=['ds', 'holiday'])
    
    # Get holiday dates
    holiday_dates = calendar_df[calendar_df['is_holiday'] == 1]['date'].tolist()
    
    holidays = []
    for date in holiday_dates:
        # Simple holiday naming based on date patterns
        holiday_name = 'Holiday'
        
        # Specific holiday naming
        if date.month == 1 and date.day == 1:
            holiday_name = 'New Year'
        elif date.month == 1 and date.day == 15:
            holiday_name = 'Mid Month Holiday'
        elif date.month == 2 and date.day == 1:
            holiday_name = 'February Holiday'
        elif date.month == 2 and date.day == 15:
            holiday_name = 'Mid February Holiday'
        elif date.month == 3 and date.day == 1:
            holiday_name = 'March Holiday'
        elif date.month == 3 and date.day == 15:
            holiday_name = 'Mid March Holiday'
        
        holidays.append({
            'ds': date,
            'holiday': holiday_name
        })
    
    holiday_df = pd.DataFrame(holidays) if holidays else pd.DataFrame(columns=['ds', 'holiday'])
    logger.info(f"Created holiday dataframe with {len(holiday_df)} holidays")
    return holiday_df


def add_features(data: pd.DataFrame, calendar_df: pd.DataFrame = None) -> pd.DataFrame:
    """Add time-based and retail-specific features for Prophet."""
    data = data.copy()
    
    # If calendar data is available, merge it
    if calendar_df is not None:
        # Merge with calendar data to get the proper features
        data_with_calendar = data.merge(
            calendar_df[['date', 'weekday', 'month', 'is_weekend', 'is_holiday', 'is_promo']], 
            left_on='ds', 
            right_on='date', 
            how='left'
        ).drop('date', axis=1)
        
        # Fill any missing values for dates not in calendar
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
        data['is_holiday'] = 0  # No holiday information available
        data['is_promo'] = 0   # No promo information available
    
    # Additional retail-specific features (as regressors)
    data['is_month_end'] = (data['ds'].dt.day >= 28).astype(int)
    data['is_month_start'] = (data['ds'].dt.day <= 3).astype(int)
    
    # Moving averages for trend analysis (not used as regressors in Prophet)
    data['sales_ma_7'] = data['y'].rolling(window=7, min_periods=1).mean()
    data['sales_ma_14'] = data['y'].rolling(window=14, min_periods=1).mean()
    
    # Ensure all regressor columns are numeric
    regressor_columns = ['is_weekend', 'is_holiday', 'is_promo', 'is_month_end', 'is_month_start']
    for col in regressor_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0).astype(int)
    
    logger.info(f"Added features. Columns: {list(data.columns)}")
    return data


@step
def preprocess_data(
    sales_data: pd.DataFrame,
    test_size: float = 0.2,
    min_train_size: int = 50,
    outlier_threshold: float = 3.0,
    enable_feature_engineering: bool = True,
) -> Tuple[
    Annotated[Dict[str, pd.DataFrame], "training_data"],
    Annotated[Dict[str, pd.DataFrame], "testing_data"],
    Annotated[List[str], "series_identifiers"],
    Annotated[pd.DataFrame, "holiday_dataframe"],
]:
    """Prepare data for forecasting with Prophet with enhanced preprocessing.

    Args:
        sales_data: Raw sales data with date, store, item, and sales columns
        test_size: Proportion of data to use for testing
        min_train_size: Minimum number of training points required
        outlier_threshold: Threshold for outlier detection (IQR multiplier)
        enable_feature_engineering: Whether to add additional features

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
        
        # Detect and handle outliers
        outliers = detect_outliers(prophet_data['y'], outlier_threshold)
        if outliers.sum() > 0:
            logger.info(f"Detected {outliers.sum()} outliers in {series_id}")
            # Cap outliers instead of removing them to preserve data
            Q1 = prophet_data['y'].quantile(0.25)
            Q3 = prophet_data['y'].quantile(0.75)
            IQR = Q3 - Q1
            upper_bound = Q3 + outlier_threshold * IQR
            lower_bound = max(0, Q1 - outlier_threshold * IQR)  # Ensure non-negative
            prophet_data.loc[outliers, 'y'] = np.clip(prophet_data.loc[outliers, 'y'], lower_bound, upper_bound)
        
        # Add features if enabled
        if enable_feature_engineering:
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
