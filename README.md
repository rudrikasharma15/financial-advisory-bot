
# 🤖📊 Streamlit Financial Advisory Bot

Welcome to your all-in-one **AI-powered financial dashboard**.  
This app blends deep learning, macroeconomic context, real-time stock insights, and conversational Gemini AI — all inside a clean Streamlit UI.

---

## 🚀 Features

| Feature                      | Description                                                                |
|------------------------------|----------------------------------------------------------------------------|
| 📊 Stock Dashboard           | Visualize and predict stock prices with trend, RSI, and risk indicators    |
| 📈 LSTM Forecasting          | Combines technical + macro data using deep learning (LSTM) models          |
| 📉 RSI & Strategy Engine     | Detects overbought/oversold conditions and offers smart suggestions        |
| 📰 Stock News Summarizer     | Fetches live news via NewsAPI + GNews for each stock                       |
| 💬 Gemini Finance Chatbot    | Ask questions like: *"Is now a good time to invest in AAPL?"*              |
| 📦 Live Portfolio Tracking   | View real-time valuation of your stock portfolio                          |
| 🎯 Savings Goal Planner      | Calculate monthly savings needed to reach your financial targets           |
| 📥 Export Tools              | Download CSV reports and view candlestick trend charts                    |
| 🔐 Secure Key Management     | API keys loaded via `.env` — safe for open-source publishing               |

---

## 🧠 Tech Stack

- **Streamlit** – Interactive web app framework
- **Gemini AI** – Conversational assistant for finance queries
- **TensorFlow + Keras** – LSTM model for price forecasting
- **FRED API** – Macroeconomic indicators: GDP, Inflation, Fed Rates
- **yFinance** – Historical stock data from Yahoo Finance
- **NewsAPI / GNews** – Real-time news aggregation
- **Python-Dotenv** – Secure API key management

---

## 📁 Project Structure

```bash
financial-dashboard/
├── streamlit_app.py         # Streamlit frontend
├── logic.py                 # LSTM, macro, news, and Gemini logic
├── .env                     # 🔐 API keys (not tracked in Git)
├── .gitignore               # Excludes .env and Python cache
├── requirements.txt         # Python dependencies
└── README.md                # You’re reading it
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone git@github.com:rudrikasharma15/financial-advisory-bot.git
cd financial-advisory-bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your API keys in a `.env` file

```ini
GEMINI_API_KEY=your_gemini_key
NEWS_API_KEY=your_newsapi_key
GNEWS_API_KEY=your_gnews_key
FRED_API_KEY=your_fred_key
```

> ✅ `.env` is automatically excluded from GitHub via `.gitignore`

---

### 4. Launch the app

```bash
streamlit run streamlit_app.py
```

---

## 🙋‍♀️ Author

**Rudrika Sharma**  
🔗 GitHub: [@rudrikasharma15](https://github.com/rudrikasharma15)

---

## 📄 License

MIT License — free for personal and commercial use with attribution.
