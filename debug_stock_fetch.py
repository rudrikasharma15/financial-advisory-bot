"""
Debug script to test stock data fetching functionality.
"""
import sys
import os
import logging
from logic import get_stock_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def test_stock_fetch():
    """Test stock data fetching with error handling."""
    print("\n=== Testing Stock Data Fetching ===")
    
    # Test with different symbols
    test_symbols = ["AAPL", "MSFT", "INVALID"]
    
    for symbol in test_symbols:
        print(f"\nFetching data for {symbol}...")
        try:
            data = get_stock_data([symbol])
            if data is not None and not data.empty:
                print(f"✅ Success! Got {len(data)} rows of data for {symbol}")
                print(f"Latest data points:\n{data.tail()}")
            else:
                print(f"❌ No data returned for {symbol}")
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {str(e)}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_stock_fetch()
