import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import os

# --- CONFIGURATION PARAMETERS ---
# Selected a diverse mix of 50+ US and Indian tickers to test pipeline scalability across different markets
TICKERS = [
    # US Tech & Finance
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'INTC',
    'AMD', 'PYPL', 'CRM', 'CSCO', 'PEP', 'KO', 'NKE', 'DIS', 'WMT', 'V',
    'MA', 'JPM', 'BAC', 'GS', 'MS', 'IBM', 'ORCL', 'UBER', 'ABNB', 'SQ',
    # Indian NSE Giants
    'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 
    'TATAMOTORS.NS', 'TATASTEEL.NS', 'WIPRO.NS', 'SBIN.NS', 'BHARTIARTL.NS',
    'ITC.NS', 'LT.NS', 'HCLTECH.NS', 'MARUTI.NS', 'SUNPHARMA.NS', 'ULTRACEMCO.NS',
    'TITAN.NS', 'BAJFINANCE.NS', 'ASIANPAINT.NS', 'POWERGRID.NS'
]

DB_NAME = 'market_data.db'
REPORT_FOLDER = 'reports'

# Ensure local directory exists for visualization outputs
if not os.path.exists(REPORT_FOLDER):
    os.makedirs(REPORT_FOLDER)

# --- DATA EXTRACTION LAYER ---
def extract_data(tickers):
    print(f"--- Initiating Batch Download for {len(tickers)} companies... ---")
    try:
        # Utilizing yfinance group download for optimized network performance (1 request vs 50)
        # Fetching 2 years of history to ensure sufficient data for 50-day MA calculation
        raw_data = yf.download(tickers, period='2y', group_by='ticker')
        print("Batch download successful.")
        return raw_data
    except Exception as e:
        print(f"Critical API Failure: {e}")
        return pd.DataFrame()

# --- DATA TRANSFORMATION LAYER ---
def transform_data(raw_data):
    print("--- Starting ETL Transformation Logic... ---")
    processed_list = []

    for ticker in TICKERS:
        try:
            # Isolating specific ticker data from MultiIndex dataframe
            df = raw_data[ticker].copy()
            
            # Step 1: Data Cleaning
            # Critical: Using forward-fill (ffill) to handle NaN values caused by 
            # different market holidays (NYSE vs NSE)
            df = df.ffill()

            # Step 2: Feature Engineering (Trend Analysis)
            # Calculating 50-Day Moving Average to identify long-term trends
            df['MA50'] = df['Close'].rolling(window=50).mean()
            
            # Step 3: Risk Metric
            # Calculating rolling standard deviation as a proxy for volatility
            df['Volatility'] = df['Close'].rolling(window=50).std()
            
            # Step 4: Normalization
            # Adding Ticker column to allow merging into a single SQL table later
            df['Ticker'] = ticker
            
            # reducing memory usage by keeping only analytical features
            df = df[['Ticker', 'Close', 'MA50', 'Volatility']]
            
            # Removing initial rows where MA50 could not be calculated (NaNs)
            processed_list.append(df.dropna())
            
        except KeyError:
            # Handling edge cases where a ticker might be delisted or renamed (e.g., TATAMOTORS)
            print(f"Integrity Check Failed: No data found for {ticker}. Skipping...")
            continue

    # consolidating all processed dataframes into a master dataset
    if processed_list:
        final_df = pd.concat(processed_list)
        print(f"ETL Transformation complete. {len(final_df)} rows ready for staging.")
        return final_df
    else:
        return pd.DataFrame()

# --- DATA LOADING LAYER (SQL) ---
def load_to_sql(df, db_name):
    print(f"--- Storing processed data in SQLite ({db_name})... ---")
    try:
        conn = sqlite3.connect(db_name)
        # Persisting data to disk. Using 'replace' to ensure fresh dataset on every run.
        df.to_sql('stock_trends', conn, if_exists='replace')
        conn.close()
        print("Database Transaction Committed Successfully.")
    except Exception as e:
        print(f"Database Error: {e}")

# --- REPORTING LAYER ---
def generate_report(df):
    print("--- Generative Visual Analytics... ---")
    
    # Selecting AAPL as a representative sample for the validation report
    example_ticker = 'AAPL'
    subset = df[df['Ticker'] == example_ticker]
    
    if not subset.empty:
        plt.figure(figsize=(12, 6))
        
        # Plotting dual-axis: Actual Price vs Calculated Trend
        plt.plot(subset.index, subset['Close'], label='Actual Price', color='blue', alpha=0.5)
        plt.plot(subset.index, subset['MA50'], label='50-Day Trend', color='red', linewidth=2)
        
        plt.title(f'Technical Analysis: {example_ticker} (Price vs Trend)')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        
        save_path = f"{REPORT_FOLDER}/{example_ticker}_analysis.png"
        plt.savefig(save_path)
        print(f"Analysis Chart exported to: {save_path}")
    else:
        print("Visualization skipped: Insufficient data for target ticker.")

# --- PIPELINE ORCHESTRATION ---
if __name__ == "__main__":
    # Execute ETL stages sequentially
    raw_data = extract_data(TICKERS)
    
    if not raw_data.empty:
        clean_df = transform_data(raw_data)
        
        if not clean_df.empty:
            load_to_sql(clean_df, DB_NAME)
            generate_report(clean_df)
            
    print("\n=== STOCK ANALYSIS PIPELINE COMPLETED ===")