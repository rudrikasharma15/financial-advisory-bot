import time
from logic import fetch_stock_data
import pandas as pd

# Test stock symbols
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL']

def test_symbol(symbol: str):
    """Fetch stock data for a symbol and report stats."""
    try:
        print(f"\n📈 Fetching data for {symbol}...")
        start_time = time.time()
        
        data = fetch_stock_data(symbol)
        elapsed = time.time() - start_time
        
        if isinstance(data, pd.DataFrame) and not data.empty:
            latest_date = data.index[-1].date()
            latest_price = data.iloc[-1, 0]  # assumes first column is price
            
            print(f"✅ {symbol} fetched in {elapsed:.2f} sec")
            print(f"   Rows: {len(data):,}, Columns: {data.shape[1]}")
            print(f"   Range: {data.index[0].date()} → {latest_date}")
            print(f"   Latest Price: ${latest_price:.2f}")
        else:
            print(f"❌ {symbol} returned empty or invalid data")

    except Exception as e:
        print(f"❌ Error fetching {symbol}: {e}")

def run_tests(symbols):
    """Run tests for multiple stock symbols."""
    print("🚀 Starting stock data fetch tests...")
    for sym in symbols:
        test_symbol(sym)
    print("\n✅ All tests completed.")

if __name__ == "__main__":
    run_tests(SYMBOLS)
