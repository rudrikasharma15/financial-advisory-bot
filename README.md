# 🤖📊 Streamlit Financial Advisory Bot

Welcome to the **Streamlit Financial Advisory Bot**, an open-source, AI-powered financial dashboard that blends deep learning, macroeconomic insights, real-time stock data, and conversational AI within a sleek Streamlit interface.
This project empowers users with financial tools and insights while welcoming contributions from the open-source community.

---

## 🌐 Live Demo
Experience the app in action with our live deployment:

[Try the Live Demo](https://financial-advisory-botgit-f7ufjegsnje6acxvg9b6ke.streamlit.app/)

---

## 🚀 Features

| Feature                  | Description |
|--------------------------|-------------|
| 📊 **Stock Dashboard**   | Visualize stock prices with trend, RSI, and risk indicators |
| 📈 **LSTM Forecasting**  | Predicts prices using deep learning (LSTM) with technical and macro data |
| 📉 **RSI & Strategy Engine** | Detects overbought/oversold conditions and provides investment suggestions |
| 📰 **Stock News Summarizer** | Fetches and summarizes live news via NewsAPI and GNews for selected stocks |
| 💬 **Gemini Finance Chatbot** | Ask financial questions like: `"Is now a good time to invest in AAPL?"` |
| 📦 **Portfolio Tracking** | View real-time valuation of your stock portfolio |
| 🎯 **Savings Goal Planner** | Calculate monthly savings needed to reach your financial goals |
| 📥 **Export Tools** | Export CSV reports and view candlestick trend charts |
| 🔐 **Secure Key Management** | Safely manage API keys via `.env` (excluded from Git) |

---

## 📸 Screenshots

Explore the app’s interface, including the stock dashboard, Gemini chatbot, and more:

[View All Screenshots](https://github.com/rudrikasharma15/financial-advisory-bot/tree/main/screenshots)

---

## 🧠 Tech Stack

- **Streamlit** – Interactive web app framework  
- **Gemini AI** – Conversational AI for financial queries  
- **TensorFlow + Keras** – LSTM models for price forecasting  
- **FRED API** – Macroeconomic data (GDP, Inflation, Fed Rates)  
- **yFinance** – Historical and real-time stock data  
- **NewsAPI / GNews** – Real-time news aggregation  
- **python-dotenv** – Secure API key management  

---

## 📁 Project Structure

```bash
financial-advisory-bot/
├── .devcontainer/           # Dev container configuration for development environments
├── .streamlit/              # Streamlit configuration (e.g., secrets.toml)
├── __pycache__/             # Python cache files (excluded via .gitignore)
├── screenshots/             # Screenshots of the app (e.g., dashboard, chatbot)
├── .DS_Store                # macOS system file (excluded via .gitignore)
├── .gitignore               # Excludes .env, Python cache, and other artifacts
├── CODE_OF_CONDUCT.md       # Contributor Covenant Code of Conduct
├── CONTRIBUTING.md          # Contribution guidelines
├── LICENSE                  # MIT License
├── README.md                # Project documentation
├── debug_stock_fetch.py     # Debugging script for stock data fetching
├── logic.py                 # LSTM, macro, news, and Gemini logic
├── requirements.txt         # Python dependencies
├── runtime.txt              # Specifies Python runtime version for deployment
├── streamlit_app.py         # Streamlit frontend
├── test_stock_fetch.py      # Unit tests for stock fetching functionality
├── yolo.txt                 # Miscellaneous file (purpose TBD)
├── yoloAchievement.txt      # Miscellaneous file (purpose TBD)
└── .env                     # 🔐 API keys (not tracked in Git)
```

---

## ⚙️ Setup Instructions

### **Prerequisites**
- Python **3.8+**
- (Recommended) Virtual environment setup:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

---

### Steps
1. 📥 Clone the Repository

```bash
     git clone https://github.com/rudrikasharma15/financial-advisory-bot.git
     cd financial-advisory-bot
```

2. 📦 Install Dependencies

```bash
     pip install -r requirements.txt
```

3. 🔑 Set Up API Keys

    Create a `.env` file in the project root:

```env
    GEMINI_API_KEY=your_gemini_key
    NEWS_API_KEY=your_newsapi_key
    GNEWS_API_KEY=your_gnews_key
    FRED_API_KEY=your_fred_key
```

4. 🚀 Run the App Locally

 ```bash
    streamlit run streamlit_app.py
```

---

## 🤝 Contributing

We welcome contributions from the open-source community!  
Especially newcomers and GSSoC'25 participants! This project is a great place to start with open-source contributions. 
See our [CONTRIBUTING.md](https://github.com/rudrikasharma15/financial-advisory-bot/blob/main/CONTRIBUTING.md) for detailed guidelines.

### Quick Start

1. Fork and Clone Your Fork:

 ```bash
     git clone https://github.com/<your-username>/financial-advisory-bot.git
     cd financial-advisory-bot
 ```

2. Create a Branch:

 ```bash
     git checkout -b feature/your-feature-name
 ```

3. Make changes, Commit and Push:

 ```bash
    git commit -m "Add your descriptive commit message"
    git push origin feature/your-feature-name
 ```

4. Submit a Pull Request: Open a pull request on GitHub, referencing any related [issues](https://github.com/rudrikasharma15/financial-advisory-bot/issues).
 
---

## 🛠 Development Setup

```bash
pip install -r requirements.txt
pip install pytest flake8 black
```

### Run tests

```bash
pytest
```

### Format code

```bash
black .
```

---

## 🛠 Troubleshooting (continued)

- **API Key Errors** – Ensure your API keys are valid and correctly formatted in .env.
- **Streamlit Not Running** – Verify Python 3.8+ and check for dependency conflicts (pip check).
- **Dependency Issues** – Use a virtual environment to isolate dependencies.

---

## 👩‍💻 Author
**Rudrika Sharma**  
🔗 [GitHub: @rudrikasharma15](https://github.com/rudrikasharma15)

---

## 📄 License
This project is licensed under the [MIT License](https://github.com/rudrikasharma15/financial-advisory-bot/blob/main/License) – feel free to use, modify, and distribute with attribution.

---

## 🌟 Acknowledgments
Thanks to the open-source community for inspiration and support.  
Powered by **Streamlit**, **TensorFlow**, and **Gemini AI**.
