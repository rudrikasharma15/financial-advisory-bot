import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow INFO/WARNING logs

import time
import logging
import requests
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from functools import lru_cache

# ML + Data
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# Define Keras aliases (so code stays readable)
Sequential = tf.keras.models.Sequential
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Bidirectional = tf.keras.layers.Bidirectional
Input = tf.keras.layers.Input
Adam = tf.keras.optimizers.Adam
EarlyStopping = tf.keras.callbacks.EarlyStopping

# Translation
from deep_translator import GoogleTranslator

# Environment
try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
except ImportError:
    pass

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Gemini setup
try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
except ImportError:
    genai = None

# ------------------ CONFIG ------------------
LOOKBACK = 30
ALERTS = {}
ALTERNATIVE_STOCKS = ["MSFT", "GOOGL", "JPM"]

# ------------------ STOCK DATA ------------------

@lru_cache(maxsize=32)
def fetch_stock_data(symbol: str, period="1y", interval="1d") -> Optional[pd.DataFrame]:
    """Fetch stock data from yfinance with caching."""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        if data.empty:
            return None
        data = data[["Close"]].rename(columns={"Close": symbol})
        data.index = data.index.tz_localize(None)
        return data
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

def get_stock_data(symbols: List[str]) -> Optional[pd.DataFrame]:
    """Fetch multiple stocks and combine into one DataFrame."""
    results = [fetch_stock_data(sym) for sym in symbols]
    valid = [df for df in results if df is not None]
    if not valid:
        return None
    return pd.concat(valid, axis=1).dropna()

# ------------------ TECHNICAL INDICATORS ------------------

def compute_rsi(series: pd.Series, periods=14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=periods).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=periods).mean()
    rs = gain / (loss.replace(0, np.nan))  # avoid div by zero
    return 100 - (100 / (1 + rs))

def add_technical_indicators(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    for sym in symbols:
        price = df[sym]
        df[f"{sym}_MA10"] = price.rolling(10).mean()
        df[f"{sym}_EMA20"] = price.ewm(span=20).mean()
        df[f"{sym}_RSI"] = compute_rsi(price, 14)
    return df

# ------------------ NEWS ------------------

NEWS_CACHE = {}
NEWS_CACHE_TTL = 300

def fetch_stock_news(symbol: str, max_articles=3) -> str:
    """Fetch latest stock news from NewsAPI."""
    if not NEWS_API_KEY:
        return "NEWS_API_KEY not configured."
    cache_key = f"{symbol}:{max_articles}"
    now = time.time()
    if cache_key in NEWS_CACHE:
        cached_time, cached_data = NEWS_CACHE[cache_key]
        if now - cached_time < NEWS_CACHE_TTL:
            return cached_data

    try:
        url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={NEWS_API_KEY}&language=en&pageSize={max_articles}"
        res = requests.get(url, timeout=5).json()
        articles = res.get("articles", [])
        if not articles:
            return f"No news for {symbol}."
        text = f"📰 Latest News for {symbol}:\n"
        for art in articles:
            title = art.get("title", "No title")
            link = art.get("url", "#")
            text += f"- {title} [Read]({link})\n"
        NEWS_CACHE[cache_key] = (now, text)
        return text
    except Exception as e:
        return f"Error fetching news: {e}"

# ------------------ ML MODEL ------------------

def prepare_model(symbols, stock_data, lookback=LOOKBACK):
    """Train a BiLSTM model for stock prediction."""
    macro = get_mock_macro_features(stock_data.index)
    combined = pd.concat([stock_data, macro], axis=1).dropna()

    target_cols = symbols
    feature_cols = [c for c in combined.columns if c not in target_cols]

    scaler_X, scaler_y = MinMaxScaler(), MinMaxScaler()
    scaled_X = scaler_X.fit_transform(combined[feature_cols])
    scaled_y = scaler_y.fit_transform(combined[target_cols])

    X, y = [], []
    for i in range(lookback, len(combined)):
        X.append(np.hstack([scaled_X[i-lookback:i], scaled_y[i-lookback:i]]))
        y.append(scaled_y[i])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.9)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dense(len(target_cols))
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    model.fit(X_train, y_train, epochs=30, batch_size=16,
              validation_split=0.1,
              callbacks=[EarlyStopping("val_loss", patience=3, restore_best_weights=True)],
              verbose=0)

    return model, scaler_X, scaler_y, combined, X_test, target_cols, y_test, split

def predict_stocks(model, scaler_y, combined, X_test, target_cols, y_test, train_size):
    preds = model.predict(X_test)
    pred_unscaled = pd.DataFrame(scaler_y.inverse_transform(preds), columns=target_cols)
    actual_unscaled = pd.DataFrame(scaler_y.inverse_transform(y_test), columns=target_cols)
    idx = combined.index[train_size+LOOKBACK:]
    pred_unscaled.index = idx
    actual_unscaled.index = idx

    evaluation = {}
    for sym in target_cols:
        evaluation[sym] = {
            "RMSE": np.sqrt(mean_squared_error(actual_unscaled[sym], pred_unscaled[sym])),
            "MAE": mean_absolute_error(actual_unscaled[sym], pred_unscaled[sym]),
            "R2": r2_score(actual_unscaled[sym], pred_unscaled[sym])
        }
    return pred_unscaled, actual_unscaled, evaluation

# ------------------ CHATBOT & UTILITIES ------------------

def get_general_financial_advice(query: str, context: str = "") -> str:
    """Use Gemini to get financial advice."""
    if not genai:
        return "Gemini API not available."
    prompt = f"Context:\n{context}\n\nUser query: {query}"
    try:
        res = genai.GenerativeModel("gemini-1.5-pro-latest").generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        return f"[Gemini Error] {e}"

def calculate_savings_goal(target_amount, years, annual_return):
    r = annual_return / 100 / 12
    n = years * 12
    monthly = (target_amount * r) / ((1 + r) ** n - 1) if r else target_amount / n
    return {"monthly_saving": abs(monthly), "years": years,
            "target_amount": target_amount, "annual_return": annual_return}

def get_mock_macro_features(dates):
    np.random.seed(42)
    return pd.DataFrame({
        "GDP_Growth": np.random.normal(2, 0.5, len(dates)),
        "Inflation": np.random.normal(2.5, 0.2, len(dates)),
        "Interest_Rate": np.random.normal(1.5, 0.3, len(dates)),
    }, index=pd.Index(dates))

def translate_response(text, lang="en"):
    return GoogleTranslator(source="auto", target=lang).translate(text)
