"""
Optimized Debug Script to Test Stock Data Fetching
"""

import sys
import logging
from typing import List, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from logic import get_stock_data

# ---------------- Logging Configuration ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------- Test Symbols ---------------- #
TEST_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "INVALID"]


# ---------------- Core Fetch Function ---------------- #
def fetch_stock(symbol: str) -> str:
    """
    Fetch stock data for a single symbol and return a result message.
    """
    try:
        data: Optional[pd.DataFrame] = get_stock_data([symbol])

        if isinstance(data, pd.DataFrame) and not data.empty:
            latest_price = data.iloc[-1, 0] if not data.empty else None
            return (
                f"✅ {symbol}: {len(data):,} rows | "
                f"Range: {data.index[0].date()} → {data.index[-1].date()} | "
                f"Latest: {latest_price:.2f}" if latest_price else f"✅ {symbol}: Data fetched"
            )
        else:
            return f"⚠️ {symbol}: No data returned"

    except Exception as e:
        return f"❌ {symbol}: Error - {str(e)}"


# ---------------- Test Runner ---------------- #
def test_stock_fetch(symbols: List[str], max_workers: int = 4) -> None:
    """
    Run stock data fetching tests in parallel for a list of symbols.
    """
    logger.info("🚀 Starting Stock Data Fetch Tests (%d symbols)", len(symbols))

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {executor.submit(fetch_stock, sym): sym for sym in symbols}

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(result)
            except Exception as e:
                logger.error("❌ %s: Unexpected error in worker - %s", symbol, str(e))

    logger.info("✅ All tests completed!")


# ---------------- Entry Point ---------------- #
if __name__ == "__main__":
    test_stock_fetch(TEST_SYMBOLS)
