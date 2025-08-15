<!-- # 🤖📊 Streamlit Financial Advisory Bot

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
Powered by **Streamlit**, **TensorFlow**, and **Gemini AI**. -->
<p align="center">
  <img src="https://github.com/Adez017/financial-advisory-bot/blob/main/screenshots/logo.png" alt="Financial Advisory Bot Logo" width="200"/>
</p>



<p align="center">
  <strong>An open-source, AI-powered financial Advisor that combines deep learning, macroeconomic insights, real-time stock data, and conversational AI within a sleek Streamlit interface.</strong>
  <br />
  <br />
  <a href="https://financial-advisory-botgit-f7ufjegsnje6acxvg9b6ke.streamlit.app/"><strong>🚀 Try Live Demo</strong></a>
  ·
  <a href="https://github.com/rudrikasharma15/financial-advisory-bot/issues"><strong>🐛 Report a Bug</strong></a>
  ·
  <a href="https://github.com/rudrikasharma15/financial-advisory-bot/issues"><strong>✨ Request a Feature</strong></a>
</p>

<p align="center">
  <a href="https://github.com/rudrikasharma15/financial-advisory-bot/stargazers"><img src="https://img.shields.io/github/stars/rudrikasharma15/financial-advisory-bot?style=for-the-badge&logo=github&color=FFDD00" alt="Stars"></a>
  <a href="https://github.com/rudrikasharma15/financial-advisory-bot/blob/main/LICENSE"><img src="https://img.shields.io/github/license/rudrikasharma15/financial-advisory-bot?style=for-the-badge&color=00BFFF" alt="License"></a>
  <a href="https://github.com/rudrikasharma15/financial-advisory-bot/network/members"><img src="https://img.shields.io/github/forks/rudrikasharma15/financial-advisory-bot?style=for-the-badge&logo=github&color=90EE90" alt="Forks"></a>
</p>

---

## 🌟 The Mission: Democratizing Financial Intelligence

The power to make informed financial decisions has traditionally been reserved for professionals with expensive tools and complex models. But what if anyone could access sophisticated financial analysis, predictive models, and AI-powered insights?

**Financial Advisory Bot** changes that.

This project provides a complete, open-source solution for intelligent financial analysis and decision-making. It's not just another stock tracker; it's a comprehensive financial intelligence platform that combines cutting-edge AI with real-world market data. Whether you're a developer building fintech solutions, an investor seeking deeper insights, or a student learning quantitative finance, this project is designed for you.

**This project welcomes GSSoC '25 contributors**, built to showcase the power of modern financial AI and open-source collaboration.

### 🔥 Core Features

*   **Advanced LSTM Forecasting:** Deep learning models that predict stock prices using both technical indicators and macroeconomic data for superior accuracy.
*   **Real-Time Intelligence:** Live stock data, news sentiment analysis, and market indicators updated in real-time for instant decision-making.
*   **Conversational AI Finance:** Powered by Gemini AI, ask complex financial questions and get intelligent, context-aware responses.
*   **Comprehensive Analytics:** RSI indicators, trend analysis, portfolio tracking, and risk assessment tools in one unified dashboard.
*   **Export & Integration Ready:** CSV exports, API integrations, and modular architecture for easy extension and integration with other systems.

---

## 🌐 Live Experience

Experience the full power of our financial intelligence platform:

**[🚀 Try the Live Demo](https://financial-advisory-botgit-f7ufjegsnje6acxvg9b6ke.streamlit.app/)**

---

## 📸 Interfaces Showcase

Explore our intuitive dashboard design and powerful analytics interface:

![Financial Advisory Bot Interfaces](https://github.com/rudrikasharma15/financial-advisory-bot/tree/main/screenshots)

---

## 🏗️ System Architecture: Financial Intelligence at Scale

The platform is architected as a modular, scalable system that seamlessly integrates multiple data sources and AI services for comprehensive financial analysis.

<details>
  <summary><strong>Click to expand the detailed data processing workflow</strong></summary>

  ### The Life of a Financial Query

  1.  **Data Ingestion Layer:** Multiple APIs (yFinance, FRED, NewsAPI, GNews) continuously fetch real-time market data, macroeconomic indicators, and news sentiment.
  2.  **Preprocessing Pipeline:**
      *   Historical stock data is normalized and technical indicators (RSI, moving averages) are calculated.
      *   Macroeconomic data (GDP, inflation, Fed rates) is synchronized with stock timelines.
      *   News articles are processed through sentiment analysis algorithms.
  3.  **LSTM Intelligence Engine:**
      *   The deep learning model processes multi-dimensional time series data.
      *   TensorFlow/Keras LSTM networks trained on both technical and fundamental indicators.
      *   Predictions are generated with confidence intervals and risk assessments.
  4.  **Conversational AI Layer:**
      *   User queries are processed by Gemini AI with financial context.
      *   Real-time market data is injected into AI responses for accuracy.
      *   Complex financial concepts are explained in accessible language.
  5.  **Visualization & Export:**
      *   Streamlit renders interactive charts, dashboards, and analytics.
      *   Users can export analysis results, generate reports, and track portfolios.
      *   All data is presented with professional-grade visualizations.

  This end-to-end pipeline ensures that every financial insight is backed by real data, advanced analytics, and intelligent AI reasoning.

</details>

---

## 🚀 The Tech Stack: Precision-Engineered for Finance

Every technology was carefully selected for its strength in financial modeling, data processing, and user experience.

| Component      | Technology                                    | Rationale & Key Benefits                                                                                                 |
| -------------- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Frontend**   | **Streamlit**                                 | **Rapid Development.** Perfect for financial dashboards with built-in widgets for data visualization, file uploads, and real-time updates. |
| **AI Engine**  | **Gemini AI**                                | **Financial Intelligence.** Advanced conversational AI specifically fine-tuned for financial queries and market analysis. |
| **ML Framework** | **TensorFlow + Keras**                      | **Deep Learning Power.** Industry-standard frameworks for building sophisticated LSTM models for time series prediction. |
| **Market Data** | **yFinance**                                | **Comprehensive Coverage.** Reliable access to historical and real-time stock data, financial statements, and market indicators. |
| **Economic Data** | **FRED API**                               | **Macro Intelligence.** Federal Reserve Economic Data for GDP, inflation, interest rates, and other economic indicators. |
| **News Intelligence** | **NewsAPI + GNews**                    | **Sentiment Analysis.** Real-time news aggregation with sentiment analysis for market-moving events and stock-specific news. |
| **Security**   | **python-dotenv**                           | **API Key Management.** Secure handling of sensitive API keys and configuration management. |

<details>
  <summary><strong>Explore the Project Architecture</strong></summary>
     
```
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
</details>

---

## 🛠️ Getting Started: From Zero to Financial Intelligence

Simple setup process designed for developers of all skill levels.

### Prerequisites

1.  **Python 3.8+:** Modern Python with async support. [Download here](https://python.org/downloads/).
2.  **Git:** For repository management. [Get it here](https://git-scm.com/).
3.  **API Keys:** You'll need free accounts for Gemini AI, NewsAPI, GNews, and FRED.

### Installation & Launch

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/rudrikasharma15/financial-advisory-bot.git
    cd financial-advisory-bot
    ```

2.  **Set Up Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    Create a `.env` file in the project root with your API credentials:
    ```env
    GEMINI_API_KEY=your_gemini_key
    NEWS_API_KEY=your_newsapi_key
    GNEWS_API_KEY=your_gnews_key
    FRED_API_KEY=your_fred_key
    ```

5.  **🎉 Launch Your Financial Intelligence Platform:**
    ```bash
    streamlit run streamlit_app.py
    ```

6.  **Access Your Dashboard:**
    *   **Main Application:** `http://localhost:8501`
    *   **Stock Analytics Dashboard**
    *   **LSTM Forecasting Engine**
    *   **Gemini AI Financial Chat**
    *   **Portfolio Tracking Tools**

---

## 💖 Join Our Community! Contributing to Financial Intelligence

We're building the future of accessible financial AI, and we need your help! This project is perfect for developers interested in fintech, AI, and open-source collaboration.

### 🌟 Why Contribute?

*   **Learn Cutting-Edge FinTech:** Work with LSTM models, real-time market data, and conversational AI.
*   **Build Your Portfolio:** Contribute to a production-ready financial platform used by real users.
*   **GSSoC '25 Friendly:** Designed specifically to welcome new contributors and open-source enthusiasts.

### Quick Start Guide

1.  **Fork & Clone:**
    ```bash
    git clone https://github.com/<your-username>/financial-advisory-bot.git
    cd financial-advisory-bot
    ```

2.  **Create Feature Branch:**
    ```bash
    git checkout -b feature/your-amazing-feature
    ```

3.  **Set Up Development Environment:**
    ```bash
    pip install -r requirements.txt
    pip install pytest flake8 black  # Development tools
    ```

4.  **Code, Test & Format:**
    ```bash
    pytest                    # Run tests
    black .                   # Format code
    flake8                    # Check style
    ```

5.  **Submit Your Contribution:**
    ```bash
    git commit -m "Add your descriptive commit message"
    git push origin feature/your-amazing-feature
    ```

📖 **Read our comprehensive [Contributing Guide](CONTRIBUTING.md)** for detailed guidelines, coding standards, and project conventions.

---

## 🛠 Development & Testing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest flake8 black

# Run the full test suite
pytest

# Format your code
black .

# Check code style
flake8
```

### Debugging & Troubleshooting

**Common Issues & Solutions:**

*   **API Key Errors:** Verify your `.env` file exists and contains valid API keys with proper formatting.
*   **Dependencies Issues:** Use a virtual environment to avoid conflicts: `python -m venv venv && source venv/bin/activate`
*   **Streamlit Not Starting:** Ensure Python 3.8+ is installed: `python --version`
*   **LSTM Model Errors:** Check TensorFlow installation: `pip install --upgrade tensorflow`

---

## 🌟 Contributors

Thanks to these amazing people who are building the future of financial AI:

<a href="https://github.com/rudrikasharma15/financial-advisory-bot/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=rudrikasharma15/financial-advisory-bot" />
</a>

---

## 👩‍💻 Project Leadership

**Rudrika Sharma** - *Creator & Lead Developer*  
🔗 [GitHub: @rudrikasharma15](https://github.com/rudrikasharma15)

---

## 📜 Code of Conduct

We're committed to fostering an inclusive, respectful community. Our project follows the Contributor Covenant Code of Conduct. In essence: **Be respectful, be collaborative, and help others learn.** Read the full [Code of Conduct](CODE_OF_CONDUCT.md) for complete guidelines.

---

## 📜 License

This project is freely available under the **MIT License**. You're welcome to use, modify, and distribute this software with proper attribution. See the [LICENSE](LICENSE) file for complete details.

---

## 🌟 Acknowledgments

Special thanks to the open-source community and the powerful technologies that make this project possible:
- **Streamlit** for the incredible app framework
- **TensorFlow** for deep learning capabilities  
- **Gemini AI** for conversational intelligence
- **Federal Reserve (FRED)** for economic data access
- **NewsAPI & GNews** for real-time market sentiment

---

<div align="center"><p>Built with ❤️ and a passion for democratizing financial intelligence. Let's make sophisticated financial analysis accessible to everyone.</p></div>

---
