# 🚀 Financial Advisory Bot - Deployment Guide

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Git** for version control
3. **API Keys** (see below)

## 🔑 Required API Keys

### 1. Google Gemini AI API Key (Required)
- Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
- Create a new API key
- Add it to your environment variables

### 2. News API Key (Optional)
- Go to [NewsAPI](https://newsapi.org/)
- Sign up for a free account
- Get your API key

## 🛠️ Local Setup

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd financial-advisory-bot
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Copy the example environment file
cp env_example.txt .env

# Edit .env with your actual API keys
# GEMINI_API_KEY=your_actual_gemini_key
# NEWS_API_KEY=your_actual_news_key
```

### 3. Run Locally
```bash
streamlit run streamlit_app.py
```

## ☁️ Deployment Options

### Option A: Streamlit Cloud (Recommended - Free)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select your repository
   - Set environment variables:
     - `GEMINI_API_KEY`: Your Gemini API key
     - `NEWS_API_KEY`: Your News API key (optional)

3. **Deploy!** Your app will be live at `https://your-app-name.streamlit.app`

### Option B: Heroku

1. **Create Heroku App**
   ```bash
   heroku create your-financial-bot
   ```

2. **Add Environment Variables**
   ```bash
   heroku config:set GEMINI_API_KEY=your_key
   heroku config:set NEWS_API_KEY=your_key
   ```

3. **Deploy**
   ```bash
   git push heroku main
   ```

### Option C: AWS/GCP/Azure

1. **Create a virtual machine**
2. **Install Python and dependencies**
3. **Set environment variables**
4. **Run with gunicorn or similar**

## 🔧 Configuration Files

### requirements.txt
All Python dependencies are listed here.

### runtime.txt
Specifies Python version for deployment platforms.

### .env
Contains your API keys (keep this file private!)

## 🚨 Important Notes

1. **Never commit .env file** - Add it to .gitignore
2. **API Rate Limits** - Be aware of API usage limits
3. **Security** - Use environment variables for sensitive data
4. **Monitoring** - Set up logging and monitoring for production

## 🐛 Troubleshooting

### Common Issues:

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **API Key Issues**
   - Check if .env file exists and has correct keys
   - Verify API keys are valid and active

3. **Port Issues**
   - Default port is 8501, change if needed

4. **Memory Issues**
   - Reduce model complexity or use smaller datasets

## 📊 Features Status

- ✅ Authentication System
- ✅ Stock Data Fetching
- ✅ Technical Analysis (RSI)
- ✅ AI Chatbot (Gemini)
- ✅ Goal Planner
- ✅ Portfolio Tracker
- ✅ SIP Calculator
- ✅ PDF Export
- ✅ Responsive UI

## 🎯 Next Steps

1. **Test all features** locally
2. **Set up monitoring** for production
3. **Add more data sources** if needed
4. **Implement user feedback** system
5. **Add more technical indicators**

## 📞 Support

If you encounter issues:
1. Check the logs
2. Verify API keys
3. Test locally first
4. Check deployment platform documentation
