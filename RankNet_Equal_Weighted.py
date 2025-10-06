# =============================================================================
# 1. IMPORTS AND CONFIGURATION
# =============================================================================
import pandas as pd
import numpy as np
import math
import time
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from scipy.stats import kendalltau
import tensorflow as tf
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import gc
import os

RAW_DATA_PATH = "WRDS-data.csv"

# Backtest period configuration
BACKTEST_START = "1980-01-01"
BACKTEST_END = "2024-12-31"  # Full period in data

# RankNet Configuration
RANKNET_EPOCHS = 100
RANKNET_EARLY_STOPPING = 10
DEFAULT_PARAMS = {
    'dropout_rate': 0.2,
    'hidden_width': 256,
    'max_gradient_norm': 1.0,
    'learning_rate': 1e-3,
    'pairs_per_batch': 1024
}

# HyperOpt Configuration
RUN_HYPEROPT = False
HYPEROPT_ITERATIONS = 25
HYPEROPT_EPOCHS = 25
HYPEROPT_EARLY_STOPPING = 10

# Data reduction options - Generate pairs for query (create pairwise dataset)
REDUCE_PAIRS = True         # Set to True to reduce number of pairs for each query
MAX_PAIRS_PER_QUERY = 2000  # Number of pairs for each query to keep if REDUCE_PAIRS is True

# Data reduction options - Load and preprocess data (for testing)
REDUCE_DATA = False         # Set to True to reduce data size for testing
SMALL_UNIVERSE_SIZE = 1000  # Number of unique PERMNOs to keep if REDUCE_DATA is True

# Random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Checkpoints Configuration
SAVE_CHECKPOINTS = True      # Whether to save intermediate results
CHECKPOINT_DIR = "./checkpoints"  # Directory for saving checkpoints
RESUME_FROM_CHECKPOINT = True # Whether to resume from checkpoints if available

# Create checkpoint directory if needed
if SAVE_CHECKPOINTS and not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
    print(f"Created checkpoint directory: {CHECKPOINT_DIR}")


# =============================================================================
# 2. CORE FINANCIAL MATH FUNCTIONS
# =============================================================================
def ewma_std(series, span=63):
    """
    Compute exponentially weighted standard deviation with specified span.
    This is a fundamental calculation for many volatility metrics.
    """
    # Convert span to alpha as per pandas ewm function
    alpha = 2 / (span + 1)

    # Calculate EWMA variance
    ewma_var = series.ewm(alpha=alpha, adjust=False).var()

    # Return square root to get standard deviation
    result = np.sqrt(ewma_var)

    # Ensure return is a scalar
    if isinstance(result, pd.Series):
        return float(result.iloc[-1]) if not result.empty else 0.01
    return float(result)

def compute_ewma(series, half_life):
    """
    Compute EWMA with half-life as specified in the paper.
    The paper uses half-life decay factors HL = log(0.5)/log(1 - 1/S)
    """
    alpha = 1 - np.exp(np.log(0.5) / half_life)
    return series.ewm(alpha=alpha, adjust=False).mean()

def compute_historical_volatility(df, span=63):
    """
    Compute historical volatility using exponentially weighted standard deviation
    with a 63-day span as specified in the paper.
    """
    if 'ret' not in df.columns:
        return 0.01  # Default value if no return data

    # Calculate EWMA standard deviation
    vol = ewma_std(df['ret'], span=span)

    # Ensure return is a scalar value, not a Series
    if isinstance(vol, pd.Series):
        return vol.iloc[-1] if len(vol) > 0 else 0.01

    return vol

def calculate_raw_return(permno, start_date, end_date, stock_histories):
    """Calculate raw return for a security between two dates."""
    if permno not in stock_histories:
        return None

    stock_data = stock_histories[permno]

    # Find prices efficiently
    start_price = None
    end_price = None

    # Get closest price to start date
    start_data = stock_data[stock_data['date'] <= start_date]
    if not start_data.empty:
        start_price = start_data.iloc[-1]['PRC']

    # Get closest price to end date
    end_data = stock_data[stock_data['date'] <= end_date]
    if not end_data.empty:
        end_price = end_data.iloc[-1]['PRC']

    if start_price is None or end_price is None or start_price <= 0:
        return None

    # Calculate raw return
    raw_return = end_price / start_price - 1.0

    return raw_return


# =============================================================================
# 3. DATA LOADING AND PREPROCESSING
# =============================================================================
def save_to_parquet_safely(df, path):
    """
    Safely save DataFrame to parquet or alternative format if parquet fails.
    """
    print(f"Preparing DataFrame for storage...")

    # First try to thoroughly clean the DataFrame
    df_clean = clean_dataframe_for_storage(df)

    # Try saving to parquet
    try:
        print(f"Attempting to save to parquet: {path}")
        df_clean.to_parquet(path)
        print(f"Successfully saved to parquet: {path}")
        return df_clean
    except Exception as e:
        print(f"Failed to save to parquet: {str(e)}")

        # Try HDF5 as an alternative
        hdf_path = path.replace('.parquet', '.h5')
        try:
            print(f"Attempting to save to HDF5: {hdf_path}")
            df_clean.to_hdf(hdf_path, key='data', mode='w')
            print(f"Successfully saved to HDF5: {hdf_path}")
            return df_clean
        except Exception as e2:
            print(f"Failed to save to HDF5: {str(e2)}")

            # As a last resort, try pickle
            pickle_path = path.replace('.parquet', '.pkl')
            try:
                print(f"Attempting to save to pickle: {pickle_path}")
                df_clean.to_pickle(pickle_path)
                print(f"Successfully saved to pickle: {pickle_path}")
                return df_clean
            except Exception as e3:
                print(f"All save attempts failed: {str(e3)}")
                raise RuntimeError("Could not save DataFrame in any format")

def clean_dataframe_for_storage(df):
    """
    Thoroughly clean a DataFrame to ensure all columns contain only scalar values.
    """
    print("Starting thorough DataFrame cleaning for storage...")
    df_clean = df.copy()

    # Process each column individually
    for col in df_clean.columns:
        print(f"Cleaning column: {col}")
        series_count = 0
        other_types = set()

        # Check the first 1000 non-null values to identify column type issues
        sample = df_clean[col].dropna().head(1000)
        for value in sample:
            if isinstance(value, pd.Series):
                series_count += 1
            elif not isinstance(value, (int, float, str, bool, pd.Timestamp, np.number)):
                other_types.add(type(value).__name__)

        if series_count > 0:
            print(f"Column {col} contains {series_count} Series objects - converting to scalars")
            # For columns with Series objects, extract the last value from each Series
            df_clean[col] = df_clean[col].apply(
                lambda x: float(x.iloc[-1]) if isinstance(x, pd.Series) else x
            )

        if other_types:
            print(f"Column {col} contains other types: {other_types} - converting to strings")
            # For columns with other complex types, convert to string
            df_clean[col] = df_clean[col].apply(
                lambda x: str(x) if not isinstance(x, (int, float, str, bool, pd.Timestamp, np.number)) else x
            )

    # Special handling for volatility columns
    for vol_col in ['Vol63', 'Vol252']:
        if vol_col in df_clean.columns:
            print(f"Special processing for {vol_col}")
            # Convert to float and handle any remaining issues
            df_clean[vol_col] = df_clean[col].apply(
                lambda x: float(x.iloc[-1]) if isinstance(x, pd.Series) else
                         (float(x) if isinstance(x, (int, float, np.number)) else 0.01)
            )

    print("DataFrame cleaning complete")
    return df_clean

def preprocess_stock_data(df):
    """
    Preprocess the entire stock dataset once to avoid redundant calculations.
    This function computes all features that don't depend on the specific month.
    """
    print("Preprocessing stock data with vectorized operations...")
    start_time = time.time()

    # Display column names for debugging
    print(f"Available columns in data: {df.columns.tolist()}")

    # Create date index dictionary for quick lookup
    date_indices = {date: i for i, date in enumerate(df['date'].unique())}

    # Sort data by PERMNO and date for efficient lookups
    df = df.sort_values(['PERMNO', 'date'])

    # Create a dictionary of stock histories for quick lookup
    print("Building stock histories lookup...")
    stock_histories = {}
    for permno, group in df.groupby('PERMNO'):
        stock_histories[permno] = group.reset_index(drop=True)

    # Get all dates for eligibility lookup
    all_dates = sorted(df['date'].unique())

    # Build eligibility lookup table using much faster vectorized approach
    eligible_stocks = build_fast_eligibility_lookup(df, all_dates)

    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

    return stock_histories, eligible_stocks, date_indices

def build_fast_eligibility_lookup(daily_data, all_dates):
    """
    Build a lookup table of eligible stocks for each date using vectorized operations.
    A stock is eligible if it has sufficient history (252 days) and price > $1.
    """
    # Check for eligibility lookup checkpoint
    eligibility_checkpoint = os.path.join(CHECKPOINT_DIR, "eligibility_lookup_exact.pkl")

    if RESUME_FROM_CHECKPOINT and os.path.exists(eligibility_checkpoint):
        print(f"Loading eligibility lookup from checkpoint: {eligibility_checkpoint}")
        return pd.read_pickle(eligibility_checkpoint)

    print("Building fast eligibility lookup table...")
    start_time = time.time()

    # Group all stocks by PERMNO
    grouped = daily_data.groupby('PERMNO')

    # Calculate lookback dates for all dates at once (more efficient)
    date_to_first_valid = {}
    for date in all_dates:
        # A stock needs at least 252 days of history to be eligible on this date
        # This follows the paper's requirement for 1-year historical data
        first_valid_date = date - pd.Timedelta(days=252)
        date_to_first_valid[date] = first_valid_date

    # Process all stocks once to determine eligibility for all dates
    eligible_stocks = {date: set() for date in all_dates}

    # Show progress
    total_stocks = len(grouped)
    print(f"Processing {total_stocks} stocks for eligibility")

    # Process in batches to provide progress updates
    batch_size = max(1, total_stocks // 10)

    for i, (permno, stock_data) in enumerate(grouped):
        if i % batch_size == 0:
            print(f"Processing eligibility: {i}/{total_stocks} stocks ({i/total_stocks*100:.1f}%)")

        # Get all dates for this stock
        stock_dates = stock_data['date'].sort_values()

        if len(stock_dates) < 252:
            # Skip stocks with insufficient history overall
            continue

        # Get first and last date for this stock
        first_date = stock_dates.iloc[0]
        last_date = stock_dates.iloc[-1]

        # Check if stock has sufficient history for each date and passes price filter
        for date in all_dates:
            if first_date > date_to_first_valid[date] or date > last_date:
                # Insufficient history or date is after stock's last date
                continue

            # Check if price > $1 at this date
            date_idx = stock_data[stock_data['date'] <= date].index
            if len(date_idx) == 0:
                continue

            latest_idx = date_idx[-1]
            latest_price = stock_data.loc[latest_idx, 'PRC']

            if abs(latest_price) > 1.0:
                eligible_stocks[date].add(permno)

    # Report stats
    avg_eligible = sum(len(stocks) for stocks in eligible_stocks.values()) / len(eligible_stocks)
    print(f"Average eligible stocks per date: {avg_eligible:.1f}")
    print(f"Eligibility lookup building completed in {time.time() - start_time:.2f} seconds")

    # Save eligibility lookup checkpoint
    if SAVE_CHECKPOINTS:
        print(f"Saving eligibility lookup to checkpoint: {eligibility_checkpoint}")
        pd.to_pickle(eligible_stocks, eligibility_checkpoint)

    return eligible_stocks

def load_and_preprocess_data():
    """Load and preprocess CRSP data, handling checkpoints if available."""
    print("Loading CRSP data...")

    # Check if preprocessed data is available as checkpoint
    data_checkpoint = os.path.join(CHECKPOINT_DIR, "preprocessed_data_exact.parquet")
    if RESUME_FROM_CHECKPOINT and os.path.exists(data_checkpoint):
        print(f"Loading preprocessed data from checkpoint: {data_checkpoint}")
        daily_data = pd.read_parquet(data_checkpoint)
    else:
        # Load raw data
        print("Loading WRDS data file...")
        daily_data = pd.read_csv(RAW_DATA_PATH, parse_dates=['date'], low_memory=False)
        print(f"Raw data loaded: {daily_data.shape[0]} records with columns: {daily_data.columns.tolist()}")

        # Data reduction for testing if enabled
        if REDUCE_DATA:
            unique_stocks = daily_data['PERMNO'].unique()[:SMALL_UNIVERSE_SIZE]
            daily_data = daily_data[daily_data['PERMNO'].isin(unique_stocks)]
            print(f"Data reduced to {len(unique_stocks)} unique stocks for testing.")

        # Filter for common stocks on NYSE as specified in paper
        # Convert columns to numeric if needed
        daily_data['SHRCD'] = pd.to_numeric(daily_data['SHRCD'], errors='coerce')
        daily_data['EXCHCD'] = pd.to_numeric(daily_data['EXCHCD'], errors='coerce')
        daily_data['PRC'] = pd.to_numeric(daily_data['PRC'], errors='coerce')

        # Apply filters from paper:
        # "actively traded firms on the NYSE from 1980 to 2019 with a CRSP share code of 10 and 11"
        # "only use stocks that are trading above $1"
        # "only consider stocks with valid prices that have been actively trading over the previous year"
        daily_data = daily_data[daily_data['SHRCD'].isin([10, 11])]
        daily_data = daily_data[daily_data['EXCHCD'] == 1]
        daily_data = daily_data[daily_data['PRC'].abs() > 1.0]
        daily_data = daily_data.sort_values('date')

        # Compute daily returns
        daily_data['RET'] = pd.to_numeric(daily_data['RET'], errors='coerce')
        daily_data['ret'] = daily_data.groupby('PERMNO')['PRC'].pct_change()
        daily_data['ret'] = daily_data['RET'].replace(0, np.nan).combine_first(daily_data['ret'])
        daily_data = daily_data.dropna(subset=['ret'])

        # Mark EOM dates - rebalancing takes place at month end as in paper
        daily_data.loc[:, 'Year'] = daily_data['date'].dt.year
        daily_data.loc[:, 'Month'] = daily_data['date'].dt.month
        daily_data.loc[:, 'EOM'] = daily_data.groupby(['Year','Month'])['date'].transform('max')

        # Save preprocessed data checkpoint
        if SAVE_CHECKPOINTS:
            print(f"Saving preprocessed data to checkpoint: {data_checkpoint}")
            daily_data.to_parquet(data_checkpoint)

    # Extract the end-of-month dates - paper rebalances monthly
    all_monthly_dates = daily_data['EOM'].drop_duplicates().sort_values().tolist()

    # Filter monthly dates based on the backtest period
    backtest_start = pd.to_datetime(BACKTEST_START)
    backtest_end = pd.to_datetime(BACKTEST_END)
    monthly_dates = [d for d in all_monthly_dates if backtest_start <= d <= backtest_end]
    print(f"Using {len(monthly_dates)} monthly sample dates between {BACKTEST_START} and {BACKTEST_END}.")

    return daily_data, monthly_dates


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
def compute_macd_features(stock_data, current_date, lookback_days=63):
    """
    Compute MACD features exactly as specified in equations (6)-(9) of the paper

    Parameters:
    stock_data - DataFrame containing stock prices
    current_date - Current date for calculation
    lookback_days - Days to look back for standardization (default 63 as in paper)

    Returns:
    Dictionary of MACD features
    """
    # Get historical data up to current date
    hist_data = stock_data[stock_data['date'] <= current_date].copy()

    if len(hist_data) < lookback_days:
        return None

    # Calculate price standard deviation over lookback period for normalization
    # This matches paper's std(p_i^(τ_m-63:τ_m)) term in equation (7)
    recent_prices = hist_data.iloc[-lookback_days:]['PRC'].values
    price_std = np.std(recent_prices)

    if price_std == 0:
        price_std = 1e-6  # Prevent division by zero

    # Calculate EWMA for all required timeframes
    # The paper uses specific short and long timescales:
    # S_k ∈ {8, 16, 32} and L_k ∈ {24, 48, 96}
    short_timescales = [8, 16, 32]
    long_timescales = [24, 48, 96]

    ewma_values = {}
    for ts in short_timescales + long_timescales:
        ewma_values[ts] = compute_ewma(pd.Series(hist_data['PRC']), half_life=ts).iloc[-1]

    # Calculate MACD indicators as per equation (8)
    # MACD(i, S, L, τ_m) = m(i, S) - m(i, L)
    macd_features = {}
    macd_normalized = {}

    for i, (s_scale, l_scale) in enumerate(zip(short_timescales, long_timescales)):
        # Calculate raw MACD
        macd = ewma_values[s_scale] - ewma_values[l_scale]
        macd_features[f'MACD_{s_scale}_{l_scale}'] = macd

        # Normalize MACD as per equation (7)
        # ξ_i^(τ_m) = MACD(i, S, L, τ_m)/std(p_i^(τ_m-63:τ_m))
        normalized_macd = macd / price_std
        macd_normalized[f'X_{s_scale}_{l_scale}'] = normalized_macd

    # Calculate composite MACD signal as per equation (9)
    # Y_i^(τ_m) = (1/3) * sum_{k=1}^3 φ(Y_i^(τ_m)(S_k, L_k))
    composite_macd = sum(macd_normalized.values()) / 3.0

    # Combine all features
    features = {
        **macd_features,
        **macd_normalized,
        'MACD_Composite': composite_macd
    }

    return features

def compute_month_features(date, stock_histories, eligible_stocks, monthly_cache, monthly_dates):
    """
    Compute features for a specific month using optimized access patterns.
    Implements feature engineering exactly as described in the paper.
    """
    eligible_permnos = eligible_stocks[date]
    if not eligible_permnos:
        return None

    # Create a list to store records for this month
    month_records = []

    # Calculate cutoff dates for return calculations
    # The paper uses 3, 6, and 12-month lookback periods
    cutoff_12m = date - pd.Timedelta(days=252)
    cutoff_6m = date - pd.Timedelta(days=126)
    cutoff_3m = date - pd.Timedelta(days=63)

    # Process each eligible stock
    for permno in eligible_permnos:
        stock_hist = stock_histories[permno]

        # Get the latest record (end of month)
        latest_idx = stock_hist[stock_hist['date'] <= date].index[-1]
        eom_record = stock_hist.loc[latest_idx].copy()

        # Find historical prices efficiently
        def find_price_at_date(cutoff):
            hist_before_cutoff = stock_hist[stock_hist['date'] <= cutoff]
            if hist_before_cutoff.empty:
                return np.nan
            return hist_before_cutoff.iloc[-1]['PRC']

        price_3m_ago = find_price_at_date(cutoff_3m)
        price_6m_ago = find_price_at_date(cutoff_6m)
        price_12m_ago = find_price_at_date(cutoff_12m)

        # Skip if any historical price is missing
        if pd.isna(price_3m_ago) or pd.isna(price_6m_ago) or pd.isna(price_12m_ago):
            continue

        # Compute raw returns exactly as in paper
        current_price = eom_record['PRC']
        ret_3m = current_price / price_3m_ago - 1.0
        ret_6m = current_price / price_6m_ago - 1.0
        ret_12m = current_price / price_12m_ago - 1.0

        # Calculate volatility for normalization
        # Paper uses a rolling exponentially weighted standard deviation with a 63-day span
        hist_data = stock_hist[stock_hist['date'] <= date].copy()

        # Calculate daily volatility
        if len(hist_data) >= 63:
            vol63 = compute_historical_volatility(hist_data.tail(63), span=63)
        else:
            vol63 = np.std(hist_data['ret']) if 'ret' in hist_data.columns else 0.01

        if len(hist_data) >= 252:
            vol252 = compute_historical_volatility(hist_data.tail(252), span=252)
        else:
            vol252 = vol63

        # Compute normalized returns as per paper methodology
        # Normalize by corresponding volatility and scale by sqrt of time
        norm_ret_3m = ret_3m / (vol63 * np.sqrt(63))
        norm_ret_6m = ret_6m / (vol252 * np.sqrt(126))
        norm_ret_12m = ret_12m / (vol252 * np.sqrt(252))

        # Calculate MACD features exactly following paper's equations (6)-(9)
        macd_features = compute_macd_features(stock_hist, date)
        if not macd_features:
            continue

        # Forward return calculation (just for labeling, not for performance measurement)
        next_month_date = date + pd.DateOffset(months=1)
        price_21d_ahead = None

        # Find the price closest to 21 trading days ahead (approximately 1 month)
        future_data = stock_hist[stock_hist['date'] >= date]
        if not future_data.empty:
            # Get data from approximately one month ahead
            future_dates = future_data['date'].tolist()
            closest_date = min(future_dates, key=lambda d: abs((d - next_month_date).days))
            if abs((closest_date - next_month_date).days) <= 10:  # Within 10 days of target
                future_idx = future_data[future_data['date'] == closest_date].index[0]
                price_21d_ahead = stock_hist.loc[future_idx, 'PRC']

        fwd21d_return = (price_21d_ahead / current_price - 1.0) if price_21d_ahead is not None else np.nan

        # Skip if forward return is missing
        if pd.isna(fwd21d_return):
            continue

        # Create record with all features
        record = {
            'PERMNO': permno,
            'date': date,
            'PRC': current_price,
            'Price_3m_ago': price_3m_ago,
            'Price_6m_ago': price_6m_ago,
            'Price_12m_ago': price_12m_ago,
            'Ret_3M': ret_3m,
            'Ret_6M': ret_6m,
            'Ret_12M': ret_12m,
            'Vol63': vol63,
            'Vol252': vol252,
            'NormRet_3M': norm_ret_3m,
            'NormRet_6M': norm_ret_6m,
            'NormRet_12M': norm_ret_12m,
            'Fwd21d_Return': fwd21d_return
        }

        # Add MACD features
        record.update(macd_features)

        # Add lagged MACD features
        for m in [1, 3, 6, 12]:
            lag_date = pd.to_datetime(date) - pd.DateOffset(months=m)
            valid_dates = [d for d in monthly_dates if d <= lag_date]
            if not valid_dates:
                continue

            chosen_date = max(valid_dates)
            if chosen_date in monthly_cache and permno in monthly_cache[chosen_date]:
                lag_features = monthly_cache[chosen_date][permno]
                record[f'X_8_24_minus{m}M'] = lag_features.get('X_8_24', 0)
                record[f'X_16_48_minus{m}M'] = lag_features.get('X_16_48', 0)
                record[f'X_32_96_minus{m}M'] = lag_features.get('X_32_96', 0)

        month_records.append(record)

    # Convert records to DataFrame
    if not month_records:
        return None

    month_data = pd.DataFrame(month_records)

    # Compute decile labels for proper ranking
    n = len(month_data)
    if n >= 10:
        # Make sure higher returns get higher deciles for proper ranking
        month_data['Rank'] = month_data['Fwd21d_Return'].rank(method='first', ascending=True)
        month_data['Decile'] = np.ceil(month_data['Rank'] / (n / 10.0)).astype(int)

        # Additional check: ensure deciles range from 1-10
        if month_data['Decile'].min() != 1 or month_data['Decile'].max() != 10:
            print(f"Unusual decile range: {month_data['Decile'].min()}-{month_data['Decile'].max()}")
    else:
        month_data['Decile'] = 5

    # Create a dictionary for this month's MACD features for future months
    month_macd_cache = {}
    for _, row in month_data.iterrows():
        month_macd_cache[row['PERMNO']] = {
            'X_8_24': row.get('X_8_24', 0),
            'X_16_48': row.get('X_16_48', 0),
            'X_32_96': row.get('X_32_96', 0)
        }

    return (date, month_data, month_macd_cache)

def process_monthly_samples(daily_data, monthly_dates):
    """Process all monthly samples using optimized methods."""
    # Define checkpoint paths
    checkpoint_file_exact = os.path.join(CHECKPOINT_DIR, "monthly_samples_exact.parquet")
    checkpoint_file_hdf = checkpoint_file_exact.replace('.parquet', '.h5')
    checkpoint_file_pkl = checkpoint_file_exact.replace('.parquet', '.pkl')

    # Check for any existing monthly samples checkpoint
    if RESUME_FROM_CHECKPOINT:
        # Try parquet first
        if os.path.exists(checkpoint_file_exact):
            try:
                print(f"Loading monthly samples from parquet: {checkpoint_file_exact}")
                monthly_samples = pd.read_parquet(checkpoint_file_exact)
                print(f"Loaded {len(monthly_samples)} records from parquet")
                return monthly_samples
            except Exception as e:
                print(f"Failed to load parquet: {str(e)}")

        # Try HDF5
        if os.path.exists(checkpoint_file_hdf):
            try:
                print(f"Loading monthly samples from HDF5: {checkpoint_file_hdf}")
                monthly_samples = pd.read_hdf(checkpoint_file_hdf, key='data')
                print(f"Loaded {len(monthly_samples)} records from HDF5")
                return monthly_samples
            except Exception as e:
                print(f"Failed to load HDF5: {str(e)}")

        # Try pickle
        if os.path.exists(checkpoint_file_pkl):
            try:
                print(f"Loading monthly samples from pickle: {checkpoint_file_pkl}")
                monthly_samples = pd.read_pickle(checkpoint_file_pkl)
                print(f"Loaded {len(monthly_samples)} records from pickle")
                return monthly_samples
            except Exception as e:
                print(f"Failed to load pickle: {str(e)}")

        # Try loading from batch files
        try:
            print("Looking for batch checkpoints...")
            # Load batch checkpoints code...
            # (remaining batch loading code is omitted for brevity)
            pass
        except Exception as e:
            print(f"Failed to load from batch files: {str(e)}")
            import traceback
            print(traceback.format_exc())

    # If no checkpoints were found or loaded successfully, process from scratch
    print(f"No usable checkpoints found. Processing {len(monthly_dates)} monthly samples from scratch...")

    # If no checkpoint, process from scratch
    print(f"Processing {len(monthly_dates)} monthly samples from scratch...")
    start_time = time.time()

    # Preprocess data once
    stock_histories, eligible_stocks, date_indices = preprocess_stock_data(daily_data)

    # Initialize cache for MACD features
    monthly_cache = {}
    monthly_samples_list = []

    # Process in chronological order because of dependencies on previous months
    batch_size = 10  # Save intermediate results every 10 months
    current_batch = []

    for i, date in enumerate(sorted(monthly_dates)):
        if i % max(1, min(10, len(monthly_dates) // 10)) == 0:
            print(f"Processing month {i+1}/{len(monthly_dates)} ({(i+1)/len(monthly_dates)*100:.1f}%)")

        result = compute_month_features(date, stock_histories, eligible_stocks, monthly_cache, monthly_dates)
        if result:
            date, month_data, month_macd_cache = result
            monthly_samples_list.append(month_data)
            current_batch.append(month_data)
            monthly_cache[date] = month_macd_cache

            # Save intermediate batch checkpoint periodically
            if len(current_batch) >= batch_size and SAVE_CHECKPOINTS:
                batch_df = pd.concat(current_batch, ignore_index=True)
                batch_path = os.path.join(CHECKPOINT_DIR, f"monthly_data_batch_exact_{len(monthly_samples_list)}.pkl")
                batch_df.to_pickle(batch_path)
                print(f"Saved intermediate batch with {len(batch_df)} records to {batch_path}")
                current_batch = []

    # Save any remaining batch
    if current_batch and SAVE_CHECKPOINTS:
        batch_df = pd.concat(current_batch, ignore_index=True)
        batch_path = os.path.join(CHECKPOINT_DIR, f"monthly_data_batch_exact_{len(monthly_samples_list)}.pkl")
        batch_df.to_pickle(batch_path)
        print(f"Saved final intermediate batch with {len(batch_df)} records to {batch_path}")

    # Clear some memory
    del eligible_stocks
    gc.collect()

    if not monthly_samples_list:
        print("No valid monthly samples were created!")
        return pd.DataFrame()

    # Combine all months
    monthly_samples = pd.concat(monthly_samples_list, ignore_index=True)

    print(f"Monthly samples processing completed in {time.time() - start_time:.2f} seconds")
    print(f"Generated {len(monthly_samples)} records across {len(monthly_samples_list)} months")

    # Save checkpoint
    if SAVE_CHECKPOINTS:
        print(f"Saving monthly samples checkpoint to: {checkpoint_file_exact}")
        monthly_samples = save_to_parquet_safely(monthly_samples, checkpoint_file_exact)

    return monthly_samples


# =============================================================================
# 5. MODEL TRAINING - RANKNET IMPLEMENTATION
# =============================================================================
def prepare_rolling_training_segments(queries, monthly_dates_sorted):
    """
    Prepare rolling training/test segments using 5-year windows as specified in the paper.
    """
    print("Preparing rolling 5-year training segments...")

    start_date = monthly_dates_sorted[0]
    end_date = monthly_dates_sorted[-1]

    def get_period_queries(start, end):
        """Get queries within a specific period"""
        period = {}
        for d in monthly_dates_sorted:
            if start <= d <= end:
                period[d] = queries[d]
        return period

    # Define rolling segments - paper mentions 5-year retuning intervals
    rolling_segments = []
    current_train_start = start_date

    while current_train_start < end_date:
        current_train_end = current_train_start + relativedelta(years=5) - timedelta(days=1)
        current_test_start = current_train_end + timedelta(days=1)
        current_test_end = current_test_start + relativedelta(years=5) - timedelta(days=1)

        # Ensure test end doesn't exceed our data range
        current_test_end = min(current_test_end, end_date)

        train_q = get_period_queries(current_train_start, current_train_end)
        test_q = get_period_queries(current_test_start, current_test_end)

        if len(train_q) == 0 or len(test_q) == 0:
            break

        rolling_segments.append((train_q, test_q))
        print(f"Segment: Train {current_train_start.date()}-{current_train_end.date()}, "
                    f"Test {current_test_start.date()}-{current_test_end.date()}")

        current_train_start = current_train_start + relativedelta(years=5)

    # Aggregate all test queries across all rolling segments
    all_test_queries = {}
    for train_q, test_q in rolling_segments:
        all_test_queries.update(test_q)

    return rolling_segments, all_test_queries

def build_ranknet_model(input_shape, hidden_width=64, dropout_rate=0.2):
    """Define RankNet Model."""
    input_features = tf.keras.Input(shape=(input_shape,), name="features")

    x = tf.keras.layers.LayerNormalization()(input_features)
    x = tf.keras.layers.Dense(hidden_width, activation='relu', name="hidden_1")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout_1")(x)

    second_hidden_width = max(1, hidden_width // 2)
    x = tf.keras.layers.Dense(second_hidden_width, activation='relu', name="hidden_2")(x)
    x = tf.keras.layers.Dropout(dropout_rate / 2, name="dropout_2")(x)

    score = tf.keras.layers.Dense(1, name="score")(x)
    model = tf.keras.Model(inputs=input_features, outputs=score, name="RankNetModel")
    return model

def generate_pairs_for_query(features, labels):
    n = len(labels)
    if n < 2:
        return [], []
    indices = np.arange(n)
    idx_i, idx_j = np.meshgrid(indices, indices)
    idx_i, idx_j = idx_i.flatten(), idx_j.flatten()

    label_i, label_j = labels[idx_i], labels[idx_j]
    mask = label_i > label_j
    pairs_i, pairs_j = idx_i[mask], idx_j[mask]

    num_pairs = len(pairs_i)

    # Random pick MAX_PAIRS_PER_QUERY pairs
    if REDUCE_PAIRS and num_pairs > MAX_PAIRS_PER_QUERY:
        sample_indices = np.random.choice(num_pairs, MAX_PAIRS_PER_QUERY, replace=False)
        pairs_i, pairs_j = pairs_i[sample_indices], pairs_j[sample_indices]

    return features[pairs_i], features[pairs_j]

def create_pairwise_dataset(queries):
    """Creates a tf.data.Dataset of pairwise examples"""
    all_features_i, all_features_j = [], []
    label_col='Decile'

    for d, df in queries.items():
        if len(df) < 2:
            continue

        feature_cols = [
            'Ret_3M', 'Ret_6M', 'Ret_12M',
            'NormRet_3M', 'NormRet_6M', 'NormRet_12M',
            'X_8_24', 'X_16_48', 'X_32_96', 'MACD_Composite'
        ] + [col for col in df.columns if 'minus' in col]

        # Ensure all feature columns exist
        for col in feature_cols[:]:
            if col not in df.columns:
                print(f"Feature column '{col}' missing in test data for {d}")
                feature_cols.remove(col)

        feat = df[feature_cols].fillna(0.0).astype(np.float32).values
        labels = df[label_col].fillna(0).astype(np.float32).values
        feat_i, feat_j = generate_pairs_for_query(feat, labels)
        if len(feat_i) > 0:
            all_features_i.append(feat_i)
            all_features_j.append(feat_j)

    features_i_np = np.concatenate(all_features_i, axis=0)
    features_j_np = np.concatenate(all_features_j, axis=0)
    labels_np = np.ones(len(features_i_np), dtype=np.float32)  # Label is always 1 for i > j
    print(f"    Generated pairs for {len(all_features_i)} queries. Total pairs created: {len(labels_np)}")

    dataset = tf.data.Dataset.from_tensor_slices(((features_i_np, features_j_np), labels_np))
    dataset = dataset.shuffle(buffer_size=len(labels_np), seed=42)
    return dataset

# RankNet training loop
def pairwise_logistic_loss(score_i, score_j):
    return tf.reduce_mean(tf.math.log1p(tf.exp(-(score_i - score_j))))

# Define search space for hyperparameters
space = {
    'dropout_rate': hp.choice('dropout_rate', [0.0, 0.2, 0.4, 0.6, 0.8]),  # architecture
    'hidden_width': hp.choice('hidden_width', [64, 128, 256, 512, 1024, 2048]),  # architecture
    'max_gradient_norm': hp.choice('max_gradient_norm', [1e-3, 1e-2, 1e-1, 1.0, 10.0]),  # optimizer
    'learning_rate': hp.choice('learning_rate', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]),  # optimizer
    'pairs_per_batch': hp.choice('pairs_per_batch', [64, 128, 256, 512, 1024])  # data prep
}

def ranknet_training_and_validation(train_dataset, valid_dataset, params, num_epoch, es_epoch=float('inf'), is_final=False):
    """
    Build, train, and valid a RankNet model (for a single segment) with provided hyperparameters and number of epoch
    Return:
    - loss (for HyperOpt for hyperparameters optimization)
    - model (for final model for prediction)
    - params (for final model for prediction and checkpoint)
    """
    # Batch pairwise dataset
    pairs_batch_size = params['pairs_per_batch']
    batched_train_dataset = train_dataset.batch(pairs_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    batched_valid_dataset = valid_dataset.batch(pairs_batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    # Build RankNet model
    params['input_shape'] = batched_train_dataset.element_spec[0][0].shape[-1]
    model = build_ranknet_model(
        input_shape=params['input_shape'],
        hidden_width=params['hidden_width'],
        dropout_rate=params['dropout_rate']
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=params['learning_rate'],
        clipnorm=params['max_gradient_norm']
    )

    # Trigger early stopping
    best_valid_loss = float('inf')  # Initialize epoch valid loss
    epoch_without_improvement = 0  # Counter to trigger early stopping

    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        # Initialize metrics to rack the average training loss and validation loss
        train_loss_metric = tf.keras.metrics.Mean()
        valid_loss_metric = tf.keras.metrics.Mean()  # to trigger early stopping

        # Training step
        for (features_i, features_j), _ in batched_train_dataset:
            with tf.GradientTape() as tape:
                score_i = model(features_i, training=True)
                score_j = model(features_j, training=True)
                loss = pairwise_logistic_loss(score_i, score_j)

            # Calculate gradients
            grads = tape.gradient(loss, model.trainable_variables)

            # Apply gradients to update weights
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Update loss metrics
            train_loss_metric.update_state(loss)
        # Compute and retrieve the average training loss as NumPy array
        epoch_train_loss = train_loss_metric.result().numpy()

        # Validation step
        for (features_i, features_j), _ in batched_valid_dataset:
            score_i = model(features_i, training=False)
            score_j = model(features_j, training=False)
            loss = pairwise_logistic_loss(score_i, score_j)

            # Update loss metrics
            valid_loss_metric.update_state(loss)
        # Compute and retrieve the average validation loss as NumPy array
        epoch_valid_loss = valid_loss_metric.result().numpy()

        # Show progress for final model training
        if is_final:
            print(f"  Epoch {epoch+1:2d}/{num_epoch} - {time.time()-epoch_start_time:5.2f}s - loss: {epoch_train_loss:.4f} - val_loss: {epoch_valid_loss:.4f}")

        # Early stopping criteria
        if epoch_valid_loss < best_valid_loss:  # validation loss improved
            best_valid_loss = epoch_valid_loss  # update best valid loss
            epoch_without_improvement = 0       # reset early stopping counter
        else:  # Validation loss did not improve
            epoch_without_improvement += 1
            if epoch_without_improvement >= es_epoch:  # trigger early stopping
                print(f"Early stopping is triggered after epoch {epoch+1}")
                break

    del batched_train_dataset, batched_valid_dataset, optimizer
    tf.keras.backend.clear_session()
    gc.collect()

    return best_valid_loss, model, params

def train_ranknet_on_segment(train_queries, valid_queries, segment_name):
    """
    Train RankNet model for a single segment using either
    (1) DEFAULT_PARAMS as default hyperparameters (fixed_params=True)
    (2) HyperOpt for hyperparameter optimization (fixed_params=False)

    If using HpyerOpt, uses 50 hyperparameter optimization iterations and specific parameter ranges.
    """
    # Define checkpoint paths
    model_weights_path = os.path.join(CHECKPOINT_DIR, f"model_{segment_name}.weights.h5")
    params_path = os.path.join(CHECKPOINT_DIR, f"params_{segment_name}.npy")

    # Check for checkpoint
    if RESUME_FROM_CHECKPOINT and os.path.exists(model_weights_path) and os.path.exists(params_path):
        print(f"Loading model and parameters from checkpoint for segment: {segment_name}")
        best_params = np.load(params_path, allow_pickle=True).item()
        model = build_ranknet_model(
            input_shape=best_params['input_shape'],
            hidden_width=best_params['hidden_width'],
            dropout_rate=best_params['dropout_rate']
        )
        model.load_weights(model_weights_path)
        return model, best_params

    # If no checkpoint available, train from scratch
    print(f"Starting RankNet training for segment: {segment_name}")
    segment_start_time = time.time()
    tf.keras.backend.clear_session()
    gc.collect()

    # Prepare training data
    print(f"  Generating training pairs from {len(train_queries)} queries")
    train_dataset = create_pairwise_dataset(train_queries)
    print(f"  Generating validation pairs from {len(valid_queries)} queries")
    valid_dataset = create_pairwise_dataset(valid_queries)

    # Hyperparameters for training RankNet
    if not RUN_HYPEROPT:
        # Using (1) DEFAULT_PARAMS as default hyperparameters
        print(f"Using default hyperparameters: {DEFAULT_PARAMS}")
        final_params = DEFAULT_PARAMS.copy()
    else:
        # Using (2) HyperOpt for hyperparameter optimization
        print(f"Running HyperOpt on segment {segment_name}")

        # Define objective function for HyperOpt
        def objective(params):
            """
            Objective function for HyperOpt: train RankNet and return validation loss
            """
            tf.keras.backend.clear_session()
            gc.collect()
            print(f"[HyperOpt] Trying Params: {params}")

            hyperopt_best_valid_loss, _, _ = ranknet_training_and_validation(
                train_dataset,
                valid_dataset,
                params=params,
                num_epoch=HYPEROPT_EPOCHS,
                es_epoch=HYPEROPT_EARLY_STOPPING,
                is_final=False)

            return {'loss': hyperopt_best_valid_loss, 'status': STATUS_OK}

        # Run hyperparameter optimization
        trials = Trials()
        best = fmin(
            objective,
            space=space,
            algo=tpe.suggest,
            max_evals=HYPEROPT_ITERATIONS,
            trials=trials,
            show_progressbar=True
        )

        # Get best hyperparameters
        best_params = {
            'dropout_rate': [0.0, 0.2, 0.4, 0.6, 0.8][best['dropout_rate']],
            'hidden_width': [64, 128, 256, 512, 1024, 2048][best['hidden_width']],
            'max_gradient_norm': [1e-3, 1e-2, 1e-1, 1.0, 10.0][best['max_gradient_norm']],
            'learning_rate': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0][best['learning_rate']],
            'pairs_per_batch': [64, 128, 256, 512, 1024][best['pairs_per_batch']]
        }
        print(f"Best hyperparameters for segment {segment_name}: {best_params}")
        final_params = best_params.copy()


    # Train final model with best hyperparameters
    print(f"Training final model for segment {segment_name}")

    _, final_model, final_params = ranknet_training_and_validation(
        train_dataset,
        valid_dataset,
        params=final_params,
        num_epoch=RANKNET_EPOCHS,
        es_epoch=RANKNET_EARLY_STOPPING,
        is_final=True)

    # Save model and parameters to checkpoint
    if SAVE_CHECKPOINTS:
        print(f"Saving model and parameters for segment {segment_name}")
        final_model.save_weights(model_weights_path)
        np.save(params_path, final_params)

    print(f"RankNet training for segment {segment_name} completed in {time.time() - segment_start_time:.2f} seconds")

    del train_dataset, valid_dataset
    gc.collect()

    return final_model, final_params

def train_and_predict_all_segments_ranknet(rolling_segments, stock_histories):
    """
    Train RankNet models on each segment and make predictions
    using either (1) default hyperparameters or (2) hyperparameter optimization.
    """
    model_scores_all = {'RankNet': {}}
    best_params_by_segment = []

    # For each rolling segment, train RankNet and predict on test queries
    for seg_idx, (train_q, test_q) in enumerate(rolling_segments):
        print(f"\nProcessing Rolling Segment {seg_idx+1}/{len(rolling_segments)} for RankNet")
        print("------------------------------------------")

        # Split train_q into train and validation sets (90/10 split)
        train_dates = sorted(train_q.keys())
        val_size = max(1, int(len(train_dates) * 0.1))
        val_dates = train_dates[-val_size:]
        train_dates = train_dates[:-val_size]
        train_q_subset = {d: train_q[d] for d in train_dates}
        val_q = {d: train_q[d] for d in val_dates}

        segment_name = f"segment_{seg_idx+1}"
        try:
            # Train RankNet model
            ranknet_model, best_params = train_ranknet_on_segment(train_q_subset, val_q, segment_name)
            best_params_by_segment.append(best_params)

            # Predict on test queries and store in global dictionary
            print(f"Making predictions for {segment_name}")
            for d, df in test_q.items():
                feature_cols = [
                    'Ret_3M', 'Ret_6M', 'Ret_12M',
                    'NormRet_3M', 'NormRet_6M', 'NormRet_12M',
                    'X_8_24', 'X_16_48', 'X_32_96', 'MACD_Composite'
                ] + [col for col in df.columns if 'minus' in col]

                # Ensure all feature columns exist
                for col in feature_cols[:]:
                    if col not in df.columns:
                        print(f"Feature column '{col}' missing in test data for {d}")
                        feature_cols.remove(col)

                feat = df[feature_cols].fillna(0.0).astype(np.float32).values
                scores = ranknet_model.predict(feat, verbose=0).flatten()
                model_scores_all['RankNet'][d] = scores
            print(f"  Score predictions for {segment_name} completed")

        except Exception as e:
            print(f"ERROR during Training for {segment_name}: {e}")

    return model_scores_all, best_params_by_segment


# =============================================================================
# 6. PERFORMANCE EVALUATION
# =============================================================================
def ndcg_at_k(y_true, y_score, k=100):
    """
    Calculate NDCG@k for a single query.
    NDCG (Normalized Discounted Cumulative Gain) is a ranking metric.

    Parameters:
    y_true: numpy array of true relevance scores
    y_score: numpy array of predicted scores
    k: number of items to consider

    Returns:
    NDCG@k value
    """
    # Limit k to the number of items
    k = min(k, len(y_true))

    # Get indices that would sort y_score in descending order
    pred_order = np.argsort(y_score)[::-1]

    # Take top k items based on predicted scores
    y_true_at_k = y_true[pred_order[:k]]

    # Calculate DCG (discounted cumulative gain)
    # Scale returns to prevent negative gains
    min_val = np.min(y_true)
    if min_val < 0:
        y_true_at_k = y_true_at_k - min_val + 1  # Make all values positive, start from 1

    gain = 2**y_true_at_k - 1
    discounts = np.log2(np.arange(len(y_true_at_k)) + 2)
    dcg = np.sum(gain / discounts)

    # Calculate ideal DCG - sort relevance scores in descending order
    ideal_order = np.argsort(y_true)[::-1]
    ideal_y_true = y_true[ideal_order[:k]]

    # Apply same scaling to ideal relevance scores
    if min_val < 0:
        ideal_y_true = ideal_y_true - min_val + 1

    ideal_gain = 2**ideal_y_true - 1
    ideal_discounts = np.log2(np.arange(len(ideal_y_true)) + 2)
    ideal_dcg = np.sum(ideal_gain / ideal_discounts)

    if ideal_dcg == 0:
        return 0

    return dcg / ideal_dcg

def compute_ranking_metrics(test_queries, model_scores_dict):
    """
    Compute ranking metrics for different models.
    Ranking metrics evaluate how well the model orders stocks.
    """
    tau_stats = {}
    ndcg_long_stats = {}
    ndcg_short_stats = {}
    corr_stats = {}  # Added to track correlations

    for model, score_map in model_scores_dict.items():
        taus = []
        ndcgs_long = []
        ndcgs_short = []
        corrs = []  # Track correlation between scores and returns

        for d, df in test_queries.items():
            if len(df) < 2 or d not in score_map:
                continue

            # Calculate Kendall's Tau
            tau, _ = kendalltau(df['Fwd21d_Return'], score_map[d])
            taus.append(tau)

            # Calculate correlation for diagnostics
            corr = np.corrcoef(df['Fwd21d_Return'], score_map[d])[0, 1]
            corrs.append(corr)

            # For NDCG calculation
            k = min(100, len(df))

            # For longs, higher scores should correspond to higher returns
            ndcg_long = ndcg_at_k(df['Fwd21d_Return'].values, score_map[d], k=k)
            ndcgs_long.append(ndcg_long)

            # For shorts, higher scores should correspond to lower returns
            # Invert returns for shorts
            ndcg_short = ndcg_at_k(-df['Fwd21d_Return'].values, score_map[d], k=k)
            ndcgs_short.append(ndcg_short)

            # Add detailed logging for first few months to diagnose
            if len(ndcgs_long) <= 3:
                top_scored_returns = df.iloc[np.argsort(score_map[d])[-10:]]['Fwd21d_Return'].values
                bottom_scored_returns = df.iloc[np.argsort(score_map[d])[:10]]['Fwd21d_Return'].values
                print(f"Month {d.strftime('%Y-%m')} NDCG diagnostics:")
                print(f"  NDCG@{k} (longs): {ndcg_long:.4f}")
                print(f"  NDCG@{k} (shorts): {ndcg_short:.4f}")
                print(f"  Top 10 scored stocks avg return: {np.mean(top_scored_returns):.6f}")
                print(f"  Bottom 10 scored stocks avg return: {np.mean(bottom_scored_returns):.6f}")

        tau_stats[model] = np.mean(taus) if taus else None
        ndcg_long_stats[model] = np.mean(ndcgs_long) if ndcgs_long else None
        ndcg_short_stats[model] = np.mean(ndcgs_short) if ndcgs_short else None
        corr_stats[model] = np.mean(corrs) if corrs else None

        # Log correlation diagnostics
        if corrs:
            positive_corrs = sum(c > 0 for c in corrs)
            print(f"{model}: {positive_corrs}/{len(corrs)} months have positive correlation")
            print(f"{model}: Average correlation = {np.mean(corrs):.4f}")

    return tau_stats, ndcg_long_stats, ndcg_short_stats, corr_stats

def compute_performance_metrics(portfolio_returns, risk_free_rate=0.0):
    """
    Compute performance metrics exactly as described in the paper.

    Note: The paper explicitly states they don't use risk-free rate in calculations.
    """
    avg_ret = portfolio_returns.mean()
    vol = portfolio_returns.std()

    # Calculate Sharpe ratio without risk-free rate as per paper
    sharpe = avg_ret / vol * np.sqrt(12) if vol > 0 else np.nan

    # Calculate downside deviation (standard deviation of negative returns only)
    downside = portfolio_returns[portfolio_returns < 0].std() if len(portfolio_returns[portfolio_returns < 0]) > 0 else 0

    # Calculate Sortino ratio without risk-free rate as per paper
    sortino = avg_ret / downside * np.sqrt(12) if downside != 0 else np.nan

    # Calculate drawdown and maximum drawdown - ensure exact method as paper
    cum = (1 + portfolio_returns).cumprod()
    drawdown = 1 - cum / cum.cummax()
    max_dd = drawdown.max()

    # Calculate Calmar ratio without risk-free rate as per paper
    calmar = avg_ret * 12 / max_dd if max_dd != 0 else np.nan

    # Calculate percentage of positive returns
    pct_positive = (portfolio_returns > 0).mean()

    # Calculate average profit / average loss ratio
    avg_profit = portfolio_returns[portfolio_returns > 0].mean() if len(portfolio_returns[portfolio_returns > 0]) > 0 else 0
    avg_loss = abs(portfolio_returns[portfolio_returns < 0].mean()) if len(portfolio_returns[portfolio_returns < 0]) > 0 else 0
    pl_ratio = avg_profit / avg_loss if avg_loss != 0 else np.nan

    return {
        'E[returns]': avg_ret,
        'Volatility': vol,
        'Sharpe': sharpe,
        'Downside Dev.': downside,
        'MDD': max_dd,
        'Sortino': sortino,
        'Calmar': calmar,
        '% Positive Returns': pct_positive,
        'Avg. P/Avg. L': pl_ratio
    }

def calculate_portfolio_correlation(long_permnos, short_permnos, current_date, stock_histories, lookback_days=126):
    """
    Calculate correlation between long and short portfolios.
    Performance-optimized version to avoid DataFrame fragmentation.
    """
    # Determine historical date range
    cutoff_date = current_date - pd.Timedelta(days=lookback_days)

    # Get all stocks we're interested in
    all_permnos = set(long_permnos) | set(short_permnos)

    # Create a dictionary to store return data by date
    date_to_returns = {}

    # First pass: collect all dates and returns
    for permno in all_permnos:
        if permno not in stock_histories:
            continue

        stock_data = stock_histories[permno]
        historical_data = stock_data[(stock_data['date'] <= current_date) &
                                    (stock_data['date'] >= cutoff_date)]

        if 'ret' not in historical_data.columns or historical_data.empty:
            continue

        # Store returns by date
        for _, row in historical_data.iterrows():
            date = row['date']
            ret = row['ret']

            if date not in date_to_returns:
                date_to_returns[date] = {}

            date_to_returns[date][permno] = ret

    if not date_to_returns:
        return 0.0  # Default correlation if no data

    # Create DataFrame directly from the collected data
    # This avoids adding columns one by one
    data = []
    for date, returns in date_to_returns.items():
        row = {'date': date}
        row.update(returns)
        data.append(row)

    if not data:
        return 0.0

    daily_returns = pd.DataFrame(data).set_index('date')

    # Fill NaN values with 0
    daily_returns = daily_returns.fillna(0)

    # Get valid permnos (those that exist in our returns data)
    valid_long_permnos = [p for p in long_permnos if p in daily_returns.columns]
    valid_short_permnos = [p for p in short_permnos if p in daily_returns.columns]

    if not valid_long_permnos or not valid_short_permnos:
        return 0.0

    # Calculate equal-weighted portfolio returns
    long_portfolio = daily_returns[valid_long_permnos].mean(axis=1)
    short_portfolio = daily_returns[valid_short_permnos].mean(axis=1)

    # Calculate correlation
    correlation = long_portfolio.corr(short_portfolio)

    # Handle NaN values
    if pd.isna(correlation):
        return 0.0

    return correlation


# =============================================================================
# 7. PORTFOLIO CONSTRUCTION
# =============================================================================
def construct_portfolio_returns(test_queries, model_scores_dict, stock_histories, all_dates):
    """
    Construct portfolio returns exactly according to paper's equation (1):

    r_{CSM}(τ_m, τ_{m+1}) = (σ^{tgt}/σ_m) * (1/n_m) * sum_{i=1}^{n_m} (X_i^{(τ_m)} * r_i(τ_m, τ_{m+1})/σ_i^{(τ_m)})

    This constructs long/short portfolios with proper volatility targeting and
    security-level risk normalization.

    ---- UPDATED IMPLEMENTATION: Instead of volatility targeting, we assign equal weights
         to all long and short positions and then apply leverage to achieve 150% gross exposure.
    """
    port_returns = {model: [] for model in model_scores_dict.keys()}
    port_dates = []

    # Add a benchmark portfolio
    benchmark_returns = []

    # Sort dates for chronological processing
    sorted_dates = sorted(test_queries.keys())

    # For diagnostics and debugging
    portfolio_stats = []

    for i, current_date in enumerate(sorted_dates):
        if i % 50 == 0:
            print(f"Processing portfolio for date {i+1}/{len(sorted_dates)}: {current_date.strftime('%Y-%m-%d')}")

        df = test_queries[current_date]

        # Find next rebalance date
        next_date_idx = i + 1
        if next_date_idx >= len(sorted_dates):
            continue

        next_date = sorted_dates[next_date_idx]
        port_dates.append(current_date)

        # Calculate universe return as benchmark
        universe_returns = []
        for permno in df['PERMNO']:
            raw_ret = calculate_raw_return(permno, current_date, next_date, stock_histories)
            if raw_ret is not None:
                universe_returns.append(raw_ret)

        if universe_returns:
            benchmark_returns.append(np.mean(universe_returns))
        else:
            benchmark_returns.append(0.0)

        # Process each model
        for model, score_map in model_scores_dict.items():
            if current_date not in score_map:
                port_returns[model].append(np.nan)
                continue

            # Add scores to dataframe
            scores = score_map[current_date]
            df_temp = df.copy()
            df_temp['score'] = scores

            # Sort by score for ranking
            df_sorted = df_temp.sort_values('score', ascending=False)

            # Select by decile as specified in paper
            # The paper states they select top and bottom deciles (10%) capped at 100 stocks
            num_stocks = len(df_sorted)
            num_decile = 100  # Fixed 100 stocks as per paper
            if len(df_sorted) < 200:  # Handle small universe edge case
                num_decile = max(int(len(df_sorted) * 0.1), 1)

            # Select top and bottom deciles
            longs = df_sorted.head(num_decile)
            shorts = df_sorted.tail(num_decile)

            # Get PERMNOs
            long_permnos = longs['PERMNO'].tolist()
            short_permnos = shorts['PERMNO'].tolist()

            # Log portfolio sizes
            if i % 50 == 0:
                print(f"  {model}: Universe size = {num_stocks}, Selected {num_decile} stocks per side")

            # ---- UPDATED IMPLEMENTATION: Equal Weighting with 150% Gross Exposure ----

            # Process long positions: compute raw returns for each long
            long_returns = []
            for permno in long_permnos:
                raw_ret = calculate_raw_return(permno, current_date, next_date, stock_histories)
                if raw_ret is not None:
                    long_returns.append(raw_ret)

            # Process short positions: compute raw returns for each short and negate them
            short_returns = []
            for permno in short_permnos:
                raw_ret = calculate_raw_return(permno, current_date, next_date, stock_histories)
                if raw_ret is not None:
                    short_returns.append(-raw_ret)

            # Check if we have valid returns for both sides
            if not long_returns or not short_returns:
                port_returns[model].append(np.nan)
                continue

            # Calculate average returns for longs and shorts
            avg_long_return = np.mean(long_returns)
            avg_short_return = np.mean(short_returns)

            # Final portfolio return: equal weighting, with 150% gross exposure
            port_return = 1.5 * (avg_long_return + avg_short_return)

            # Log diagnostics
            if i % 50 == 0:
                print(f"  Long avg return: {avg_long_return:.6f}, Short avg return: {avg_short_return:.6f}")
                print(f"  Final portfolio return (150% exposure): {port_return:.6f}")

            # Save statistics for analysis (optional)
            portfolio_stats.append({
                'date': current_date,
                'model': model,
                'universe_size': num_stocks,
                'decile_size': num_decile,
                'long_count': len(long_returns),
                'short_count': len(short_returns),
                'long_avg_return': avg_long_return,
                'short_avg_return': avg_short_return,
                'scaled_return': port_return
            })

            # Add to portfolio returns
            port_returns[model].append(port_return)

    # Create dataframe of portfolio returns
    port_df = pd.DataFrame(port_returns, index=pd.to_datetime(port_dates))

    # Add benchmark returns
    port_df['Benchmark'] = benchmark_returns

    # Replace NaN values using forward fill then backward fill
    port_df = port_df.ffill().bfill()

    # Calculate cumulative returns
    cumulative = (1 + port_df).cumprod() - 1

    # Save portfolio statistics for analysis
    if SAVE_CHECKPOINTS:
        stats_df = pd.DataFrame(portfolio_stats)
        stats_path = os.path.join(CHECKPOINT_DIR, "portfolio_stats_exact.pkl")
        stats_df.to_pickle(stats_path)
        print(f"Saved portfolio statistics to {stats_path}")

    return port_df, cumulative


# =============================================================================
# 8. RESULT COMPARISON
# =============================================================================




# =============================================================================
# 9. MAIN EXECUTION
# =============================================================================
def main():
    """Main execution function."""
    total_start_time = time.time()
    print("Starting RankNet Implementation")
    print(f"Backtest period: {BACKTEST_START} to {BACKTEST_END}")

    # Step 1: Load and preprocess data
    daily_data, monthly_dates = load_and_preprocess_data()

    # Step 2: Feature engineering and monthly sampling
    monthly_samples = process_monthly_samples(daily_data, monthly_dates)

    # Construct queries dictionary from the results
    queries = {}
    for date, group in monthly_samples.groupby('date'):
        queries[date] = group.reset_index(drop=True)

    # Step 3: Get stock histories for return calculation
    stock_histories, _, _ = preprocess_stock_data(daily_data)

    # Step 4: Rolling segments
    monthly_dates_sorted = sorted(queries.keys())
    rolling_segments, all_test_queries = prepare_rolling_training_segments(queries, monthly_dates_sorted)

    # Step 5: Train models and make predictions
    model_scores_all, best_params_by_segment = train_and_predict_all_segments_ranknet(rolling_segments, stock_histories)

    # Step 6: Construct portfolios and evaluate performance
    print("Constructing portfolios and calculating returns...")
    full_port_df, full_cumulative = construct_portfolio_returns(all_test_queries, model_scores_all, stock_histories, monthly_dates_sorted)

    # Calculate without risk-free rate as per paper
    risk_free_rate = 0.0  # Paper explicitly mentions not using risk-free rate

    # Compute performance metrics
    print("Computing performance metrics...")
    model_metrics = {}
    for model in full_port_df.columns:
        metrics = compute_performance_metrics(full_port_df[model], risk_free_rate)
        model_metrics[model] = metrics
        print(f"{model} performance metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")

    # Step 7: Compute ranking metrics
    print("Computing ranking metrics...")
    tau_stats, ndcg_long_stats, ndcg_short_stats, corr_stats = compute_ranking_metrics(all_test_queries, model_scores_all)
    for model in tau_stats:
        print(f"{model} ranking metrics:")
        print(f"  Kendall Tau: {tau_stats[model]:.4f}")
        print(f"  NDCG@100 (longs): {ndcg_long_stats[model]:.4f}")
        print(f"  NDCG@100 (shorts): {ndcg_short_stats[model]:.4f}")
        print(f"  Return-Score Correlation: {corr_stats[model]:.4f}")

if __name__ == "__main__":
    main()