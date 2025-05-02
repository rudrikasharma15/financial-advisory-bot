
import pandas as pd
import numpy as np
import time
import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from googletrans import Translator
from tensorflow.keras.layers import Bidirectional

import google.generativeai as genai

import os
from dotenv import load_dotenv
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def get_general_financial_advice(query):
    try:
        # Use the latest supported Gemini model
        model = genai.GenerativeModel("gemini-1.5-pro-latest")

        # Call Gemini with the full question
        response = model.generate_content(query)

        return response.text
    except Exception as e:
        return f"Error getting response: {e}"

# ------------------ CONFIG ------------------
START_DATE = '2015-01-01'
LOOKBACK = 30  # Reduced to ensure sufficient data
TRAIN_SPLIT = 0.9
ALERTS = {}
PORTFOLIO = {}
TRANSLATOR = Translator()
ALTERNATIVE_STOCKS = ['MSFT', 'GOOGL', 'JPM']

# ------------------ DATA FETCH ------------------

def fetch_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period='60d', interval='1h')  # More granularity and fresher
        if data.empty:
            print(f"Error: No data returned for {symbol}.")
            return None
        data = data[['Close']].rename(columns={'Close': symbol})
        data.index = data.index.tz_localize(None)  # Remove timezone
        if len(data) < LOOKBACK:
            print(f"Error: Insufficient data for {symbol}. Only {len(data)} days available, need at least {LOOKBACK}.")
            return None
        print(f"Successfully fetched {symbol} with {len(data)} days of data")
        return data
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def get_stock_data(symbols, lookback=30):
    all_data = []
    for symbol in symbols:
        print(f"Fetching {symbol}...")
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period='2y', interval='1d')
            if data.empty:
                print(f"[Error] No data for {symbol}.")
                continue
            data = data[['Close']].rename(columns={'Close': symbol})
            data.index = data.index.tz_localize(None)
            if len(data) < lookback:
                print(f"[Error] Not enough data for {symbol} (have {len(data)}, need {lookback}).")
                continue
            print(f"[OK] Fetched {symbol} with {len(data)} rows.")
            all_data.append(data)
        except Exception as e:
            print(f"[Exception] Failed to fetch {symbol}: {e}")
    if not all_data:
        print("[Error] No valid data fetched.")
        return None
    return pd.concat(all_data, axis=1).dropna()


# ------------------ NEWS FETCH ------------------

import requests

def fetch_stock_news(symbol, max_articles=3):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={API_KEY}&language=en"
    response = requests.get(url)
    articles = response.json().get("articles", [])[:max_articles]

    if not articles:
        return f"No recent news found for {symbol}."

    news_str = f"📰 **Latest News for {symbol}**:\n"
    for article in articles:
        title = article['title']
        link = article['url']
        source = article['source']['name']
        date = article['publishedAt'][:10]
        news_str += f"- **{title}** ({source}, {date})\n  [Read more]({link})\n"
    return news_str

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

LOOKBACK = 30

def add_technical_indicators(df, symbols):
    for symbol in symbols:
        price = df[symbol]

        # Moving Averages
        df[f'{symbol}_MA10'] = price.rolling(window=10).mean()
        df[f'{symbol}_EMA20'] = price.ewm(span=20, adjust=False).mean()

        # RSI
        df[f'{symbol}_RSI'] = compute_rsi(price, 14)

        # MACD
        ema12 = price.ewm(span=12, adjust=False).mean()
        ema26 = price.ewm(span=26, adjust=False).mean()
        df[f'{symbol}_MACD'] = ema12 - ema26

        # Bollinger Bands
        df[f'{symbol}_Bollinger_Upper'] = price.rolling(20).mean() + price.rolling(20).std() * 2
        df[f'{symbol}_Bollinger_Lower'] = price.rolling(20).mean() - price.rolling(20).std() * 2

        # ✅ New Features:
        df[f'{symbol}_Momentum'] = price - price.shift(10)
        df[f'{symbol}_Volatility'] = price.rolling(10).std()

        if 'Volume' in df.columns:
            df[f'{symbol}_Volume_MA'] = df['Volume'].rolling(10).mean()

    return df


def compute_rsi(series, periods=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

    print("Stock index preview:", self.stock_data.index[:5])
    print("Index type:", type(self.stock_data.index))

def get_mock_macro_features(dates):
    np.random.seed(42)

    # Ensure it's a proper list of timestamps
    if not isinstance(dates, (pd.Index, list, np.ndarray)):
        try:
            dates = list(dates)
        except Exception as e:
            print(f"[Error] Invalid dates passed to get_mock_macro_features: {e}")
            dates = pd.date_range(start='2020-01-01', periods=100)  # fallback

    dates = pd.Index(dates)

    return pd.DataFrame({
        'GDP_Growth': np.random.normal(2, 0.5, len(dates)),
        'Inflation': np.random.normal(2.5, 0.2, len(dates)),
        'Interest_Rate': np.random.normal(1.5, 0.3, len(dates))
    }, index=dates)




def create_dataset(dataset, target_cols, step):
    X, y = [], []
    for i in range(step, len(dataset)):
        X.append(dataset.iloc[i-step:i].values)
        y.append([dataset.iloc[i, dataset.columns.get_loc(col)] for col in target_cols])
    return np.array(X), np.array(y)


def predict_stocks(model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size):
    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    print("[OK] Predicting...")
    predictions = model.predict(X_test)

    if predictions.shape != y_test.shape:
        raise ValueError("Prediction shape mismatch!")

    # Inverse transform predictions and actuals to get original prices
    pred_unscaled = pd.DataFrame(
        scaler_y.inverse_transform(predictions),
        columns=target_cols
    )

    actual_unscaled = pd.DataFrame(
        scaler_y.inverse_transform(y_test),
        columns=target_cols
    )

    # Ensure proper index alignment
    test_data_index = combined_scaled.index[train_size + LOOKBACK:]
    pred_unscaled.index = test_data_index
    actual_unscaled.index = test_data_index

    results = {}
    evaluation = {}

    for symbol in target_cols:
        predicted = pred_unscaled[symbol]
        actual = actual_unscaled[symbol]

        results[symbol] = {
            "predicted": predicted,
            "actual": actual
        }

        evaluation[symbol] = {
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "MAE": mean_absolute_error(actual, predicted),
            "R2": r2_score(actual, predicted)
        }

    print("[OK] Prediction complete.")
    return results, evaluation


def prepare_model(symbols, stock_data, macro, lookback=30):
    print("Starting enhanced model preparation...")
    if stock_data is None or stock_data.empty or macro is None or macro.empty:
        print("[Error] Missing stock or macro data.")
        return None

    combined = pd.concat([stock_data, macro], axis=1).dropna()

    target_cols = symbols
    feature_cols = [col for col in combined.columns if col not in target_cols]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    train_split = int(len(combined) * 0.9)

    scaled_X = pd.DataFrame(
        scaler_X.fit_transform(combined[feature_cols]),
        columns=feature_cols,
        index=combined.index
    )
    scaled_y = pd.DataFrame(
        scaler_y.fit_transform(combined[target_cols]),
        columns=target_cols,
        index=combined.index
    )

    scaled_combined = pd.concat([scaled_X, scaled_y], axis=1)

    X, y = [], []
    for i in range(lookback, len(scaled_combined)):
        X.append(scaled_combined.iloc[i-lookback:i].values)
        y.append(scaled_y.iloc[i].values)

    X, y = np.array(X), np.array(y)

    split = int(0.9 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    from tensorflow.keras import Input
    model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(target_cols))
])



    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')

    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
        verbose=0
    )

    print("[OK] Enhanced model trained.")
    return model, scaler_X, scaler_y, scaled_combined, X_test, target_cols, y_test, split

def evaluate_predictions(model, scaler_X, scaler_y, combined_scaled, X_test, target_cols, y_test, train_size):
    predictions = model.predict(X_test)
    pred_unscaled = pd.DataFrame(scaler_y.inverse_transform(predictions), columns=target_cols)
    actual_unscaled = pd.DataFrame(scaler_y.inverse_transform(y_test), columns=target_cols)

    test_data_index = combined_scaled.index[train_size + LOOKBACK:]
    pred_unscaled.index = test_data_index
    actual_unscaled.index = test_data_index

    results = {}
    evaluation = {}

    for symbol in target_cols:
        predicted = pred_unscaled[symbol]
        actual = actual_unscaled[symbol]
        results[symbol] = {
            "predicted": predicted,
            "actual": actual
        }
        evaluation[symbol] = {
            "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
            "MAE": mean_absolute_error(actual, predicted),
            "R2": r2_score(actual, predicted)
        }

    print("[OK] Evaluation complete.")
    return results, evaluation

# ------------------ CHATBOT FEATURES ------------------
def get_advice(series):
    recent = series.dropna().iloc[-10:]
    if len(recent) < 2:
        return "Insufficient data to generate advice."

    trend = "Uptrend" if recent.iloc[-1] > recent.iloc[0] else "Downtrend"
    if trend == "Uptrend":
        return "Uptrend predicted. Consider holding or buying more."
    else:
        return "Downtrend predicted. You might want to wait."


def calculate_risk(symbol, stock_data, results):
    data = stock_data[symbol].dropna()
    if len(data) < 10:
        print(f"Warning: Insufficient data for risk calculation for {symbol}. Defaulting to 50.")
        return 50
    returns = data.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100
    rsi = stock_data[f'{symbol}_RSI'].iloc[-1]
    rsi_risk = max(0, abs(rsi - 50) - 20) / 0.7
    pred_change = (results[symbol]['predicted'].iloc[-1] - results[symbol]['actual'].iloc[-1]) / results[symbol]['actual'].iloc[-1] * 100
    pred_risk = abs(pred_change) / 2
    risk_score = min(100, (volatility + rsi_risk + pred_risk) / 3)
    print(f"Calculated risk for {symbol}: {risk_score:.2f}")
    return risk_score

def fetch_news(symbol):
    return f"News for {symbol}: [Mock] Positive earnings report released today."

def get_metrics(symbol, data):
    return f"{symbol} Metrics: MA10={data[f'{symbol}_MA10'].iloc[-1]:.2f}, RSI={data[f'{symbol}_RSI'].iloc[-1]:.2f}"

def get_strategy(advice, risk_score):
    if risk_score > 70:
        return "High-risk strategy: Avoid or use options hedging."
    elif "invest" in advice:
        return "Dollar-cost averaging"
    else:
        return "Wait and monitor"

def check_alerts(results):
    for symbol, threshold in ALERTS.items():
        if symbol in results:
            pred_change = (results[symbol]['predicted'].iloc[-1] - results[symbol]['actual'].iloc[-1]) / results[symbol]['actual'].iloc[-1]
            if abs(pred_change) > threshold:
                return f"Alert: {symbol} changed by {pred_change*100:.2f}%!"
    return "No alerts triggered."

def translate_response(text, lang='en'):
    return TRANSLATOR.translate(text, dest=lang).text

def get_alternative_options(symbol, results, stock_data):
    alternatives = {}
    for alt_symbol in ALTERNATIVE_STOCKS:
        if alt_symbol not in results:
            alt_data = fetch_stock_data(alt_symbol)
            if alt_data is not None:
                alt_rsi = compute_rsi(alt_data[alt_symbol]).iloc[-1]
                alt_trend = "Uptrend" if alt_data[alt_symbol].iloc[-1] > alt_data[alt_symbol].iloc[-10] else "Downtrend"
                alternatives[alt_symbol] = {'RSI': alt_rsi, 'Trend': alt_trend}
    if not alternatives:
        return "No viable alternatives available."
    best_alt = min(alternatives.items(), key=lambda x: abs(x[1]['RSI'] - 50))
    return f"Consider {best_alt[0]}: RSI={best_alt[1]['RSI']:.2f}, Trend={best_alt[1]['Trend']}"

EDUCATION = {
    "rsi": "RSI (Relative Strength Index) measures momentum on a scale of 0-100, indicating overbought (>70) or oversold (<30) conditions.",
    "p/e": "P/E (Price-to-Earnings) ratio compares a company's stock price to its earnings per share, useful for valuation."
}

import matplotlib.pyplot as plt
# Financial Education Section
finance_questions = [
            "What is an emergency fund?",
            "How does budgeting work?",
            "Explain the debt snowball method",
            "What is compound interest?",
            "What are ETFs?",
            "What is diversification?",
            "How does a Roth IRA work?",
            "What is a 401k match?",
            "What is an index fund?"
        ]


def handle_edu_query(self):
        query = self.edu_dropdown.value if self.edu_dropdown.value != "Select a topic..." else self.edu_input.value.strip()
        if not query:
                print("Please select a topic OR type a custom question.")
        return
        print(get_general_financial_advice(query))

def handle_market_news(self):
            print(fetch_stock_news("SPY"))

def handle_goal_calc(self):
        target = self.goal_amount.value
        years = self.goal_years.value
        ret = self.goal_return.value
        result = calculate_savings_goal(target, years, ret)
        print(f"🌟 To reach ₹{result['target_amount']:.2f} in {result['years']} years at {result['annual_return']}% return:")
        print(f"💰 Save ₹{result['monthly_saving']:.2f} per month.")

def start_chatbot(self, b):
        with self.output_area:
            print("Initializing chatbot...")
            if not symbols:
                print("Please enter at least one valid stock symbol.")
                return
            self.symbols = symbols
            print(f"Symbols: {symbols}")
            self.stock_data = get_stock_data(symbols)
            if self.stock_data is None:
                print("Stock data fetch failed.")
                return
            self.stock_data = add_technical_indicators(self.stock_data, symbols)
            macro = get_mock_macro_features(self.stock_data.index)

        try:
            model_result = prepare_model(symbols, self.stock_data, macro)
        except Exception as e:
            with self.output_area:
                print(f"[Error] Model preparation failed: {e}")
            return

        if model_result is None:
            with self.output_area:
                print("Model preparation failed.")
            return

        self.model, self.scaler_X, self.scaler_y, self.combined_scaled, \
        self.X_test, self.target_cols, self.y_test, self.train_size = model_result

        self.results, self.evaluation = predict_stocks(
            self.model, self.scaler_X, self.scaler_y, self.combined_scaled,
            self.X_test, self.target_cols, self.y_test, self.train_size
        )

        with self.output_area:
            print("Model evaluation:")
            for symbol, metrics in self.evaluation.items():
                print(f"{symbol}: RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, R²={metrics['R2']:.2f}")

def show_plot(self, b):
        with self.output_area:
            print("Available prediction results:", list(self.results.keys()))
            for symbol in self.symbols:
                if symbol not in self.results:
                    print(f"[Warning] No prediction available for {symbol}. Skipping.")
                    continue
                predicted = self.results[symbol]['predicted']
                actual = self.results[symbol]['actual']
                plt.figure(figsize=(12, 6))
                plt.plot(actual.index, actual * 100, label='Actual Return (%)', color='green')
                plt.plot(predicted.index, predicted * 100, label='Predicted Return (%)', linestyle='--', color='orange')
                plt.axhline(0, color='gray', linestyle='--', linewidth=1)
                plt.title(f'{symbol} - Predicted vs Actual 1-Day Returns')
                plt.ylabel('Return (%)')
                plt.xlabel('Date')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

def process_input(self, b):
        with self.output_area:
            if not query:
                print("Please enter a query.")
                return

            symbol = self.symbols[0] if self.symbols else "AAPL"

            if "predict" in query.lower():
                if symbol not in self.results:
                    print(f"No prediction results available for {symbol}.")
                    return
                print(f"📈 Forecasting returns for {symbol}...")
                predicted_returns = self.results[symbol]['predicted']
                latest_return = predicted_returns.iloc[-1]
                if latest_return > 0.01:
                    decision = "📈 Consider BUYING — model expects gains."
                elif latest_return < -0.01:
                    decision = "📉 Consider SELLING — model expects losses."
                else:
                    decision = "➖ HOLD — model expects little movement."
                print(f"📊 Predicted 1-day return: {latest_return*100:.2f}%")
                print(decision)
                self.show_plot(None)
                return

            elif "advice" in query.lower():
                for sym in self.symbols:
                    if sym in self.results:
                        pred_series = self.results[sym]['predicted']
                        advice = get_advice(pred_series)
                        print(f"📈 Advice for {sym}: {advice}")
                    else:
                        print(f"No prediction data available for {sym}.")
                return

            elif "sell" in query.lower():
                for symbol in self.symbols:
                    if symbol not in self.results:
                        print(f"No prediction results available for {symbol}.")
                        continue
                    pred_price = self.results[symbol]['predicted'].iloc[-1]
                    actual_price = self.results[symbol]['actual'].iloc[-1]
                    rsi = self.stock_data[f'{symbol}_RSI'].iloc[-1]
                    trend = "rising" if pred_price > actual_price else "falling"
                    print(f"🤔 {symbol} Sell Check:")
                    print(f"  - RSI: {rsi:.2f}")
                    print(f"  - Trend: {trend}")
                    print(f"  - Predicted Return: {pred_price:.4f}")
                    print(f"  - Actual Return: {actual_price:.4f}")
                    if rsi > 70 or trend == "falling":
                        print("⚠️ Consider selling or reducing exposure.")
                    else:
                        print("✅ Holding may be appropriate.")
                return

            else:
                  print(f"💬 Query: {query}")
                  matched_symbols = [sym for sym in self.symbols if sym in query.upper()]
                  symbol = matched_symbols[0] if matched_symbols else self.symbols[0]

                  try:
                      predicted = self.results[symbol]['predicted'].iloc[-1]
                      actual = self.results[symbol]['actual'].iloc[-1]
                      rsi = self.stock_data[f'{symbol}_RSI'].iloc[-1]
                      trend = "rising" if predicted > actual else "falling" if predicted < actual else "stable"

                      context = (
                          f"{symbol} predicted price: ₹{predicted:.2f}, "
                          f"actual price: ₹{actual:.2f}, RSI: {rsi:.2f}, trend: {trend}."
                      )
                      prompt = f"The user asked: '{query}'. Here is the market context: {context} What should the user do?"

                      print("🧠 Gemini is analyzing...")
                      response = get_general_financial_advice(prompt)
                      print(f"🧠 Gemini: {response}")

                  except Exception as e:
                      print("Gemini error:", e)

# ------------------ GENERAL FINANCIAL ADVICE MODULE ------------------

GENERAL_EDUCATION = {
    "emergency fund": "An emergency fund is savings for unexpected expenses like job loss or medical emergencies. Aim for 3-6 months of living expenses.",
    "budgeting": "Budgeting involves tracking income and expenses to control spending. The 50/30/20 rule is popular: 50% needs, 30% wants, 20% savings.",
    "debt snowball": "The debt snowball method means paying off your smallest debts first while making minimum payments on larger debts, gaining momentum.",
    "compound interest": "Compound interest means your money earns interest on interest. Early investing benefits from exponential growth.",
    "etf": "ETFs (Exchange Traded Funds) are collections of stocks or bonds you can buy in a single fund, offering diversification at low cost.",
    "diversification": "Diversification reduces risk by spreading investments across different assets like stocks, bonds, real estate, etc.",
    "roth ira": "A Roth IRA allows post-tax contributions and tax-free withdrawals in retirement, ideal if you expect to be in a higher tax bracket later.",
    "401k match": "A 401(k) employer match is free money. Contribute at least enough to get the full match—it’s an instant 100% return.",
    "index fund": "Index funds track a market index (like the S&P 500). They offer broad diversification and low fees, great for long-term growth."
}

def get_general_financial_advice(query, symbols=None, stock_data=None, results=None):
    import google.generativeai as genai
    import os
    from dotenv import load_dotenv
    load_dotenv()

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


    model = genai.GenerativeModel("gemini-1.5-pro-latest")

    context = ""

    if symbols and stock_data is not None and results is not None:
        for symbol in symbols:
            try:
                rsi = stock_data[f"{symbol}_RSI"].iloc[-1]
                actual = results[symbol]["actual"].iloc[-1]
                predicted = results[symbol]["predicted"].iloc[-1]
                trend = "rising" if predicted > actual else "falling"
                context += f"{symbol}: RSI = {rsi:.2f}, Trend = {trend}, Predicted Price = ₹{predicted:.2f}, Actual Price = ₹{actual:.2f}\n"
            except:
                continue

    prompt = f"""You are a financial assistant. Here's the current market context:\n{context}\n\nUser query: {query}
Provide smart, actionable, and responsible investment guidance. Avoid financial guarantees. Be brief and specific when possible."""

    try:
        result = model.generate_content(prompt)
        return result.text.strip()
    except Exception as e:
        return f"[Gemini Error] {e}"

def calculate_savings_goal(target_amount, years, annual_return_percent):
    """
    Calculate monthly saving needed to reach a financial goal.
    Args:
        target_amount (float): Desired final amount
        years (float): Years to save
        annual_return_percent (float): Expected annual return (e.g., 7%)

    Returns:
        dict with required monthly savings and projected value
    """
    r = annual_return_percent / 100 / 12  # monthly rate
    n = years * 12  # total months
    if r == 0:
        monthly = target_amount / n
    else:
        monthly = target_amount * r / ((1 + r) ** n - 1)
    monthly = abs(monthly)

    return {
        "monthly_saving": monthly,
        "years": years,
        "target_amount": target_amount,
        "annual_return": annual_return_percent
    }

