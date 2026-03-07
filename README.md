# 🐦 Twitter Sentiment Analyzer - End-to-End ML Project

A complete machine learning application that analyzes sentiment in text using NLP.

## 🌐 Live Demo
**🔗 Try it here:** https://huggingface.co/spaces/PrajwalDhoke/twitter_sentiment_analyzer

## 📋 Project Overview
This capstone project demonstrates the complete ML deployment pipeline, from model training to cloud deployment. The application classifies text as Positive, Negative, or Neutral with 98% accuracy.

## ✨ Features
- **Sentiment Classification:** Analyzes text sentiment in real-time
- **High Accuracy:** 98%+ on test dataset
- **REST API:** FastAPI backend with /health and /predict endpoints
- **Interactive UI:** Streamlit frontend with live predictions
- **Containerized:** Docker-ready application
- **Cloud Deployed:** Hosted on Hugging Face Spaces

## 🛠️ Tech Stack
**Machine Learning:**
- scikit-learn (Logistic Regression)
- TF-IDF Feature Extraction
- joblib for serialization

**Backend:**
- FastAPI
- Uvicorn (ASGI server)
- Pydantic (validation)

**Frontend:**
- Streamlit

**Deployment:**
- Docker
- Hugging Face Spaces

## 📁 Project Structure
```
twitter-sentiment-analyzer/
├── model/                  # Trained model files
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
├── app/                    # FastAPI backend
│   ├── main.py
│   └── schemas.py
├── frontend/               # Streamlit UI
│   └── app.py
├── data/                   # Training data
│   └── tweets.csv
├── Dockerfile              # Container config
├── requirements.txt        # Dependencies
├── train_model.py          # Model training
├── run_api.py             # API launcher
└── run_frontend.py        # Frontend launcher
```

## 🚀 Local Installation

### Prerequisites
- Python 3.12+
- pip
- Virtual environment (recommended)

### Setup Steps
```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/twitter-sentiment-analyzer.git
cd twitter-sentiment-analyzer

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🏃 Running Locally

### Option 1: Run API + Frontend Separately

**Terminal 1 - Start API:**
```bash
python run_api.py
```
API will run at: http://localhost:8000
API docs at: http://localhost:8000/

**Terminal 2 - Start Frontend:**
```bash
python run_frontend.py
```
Frontend will run at: http://localhost:8501

### Option 2: Run with Docker
```bash
# Build Docker image
docker build -t sentiment-analyzer .

# Run container
docker run -p 8000:8000 -p 8501:8501 sentiment-analyzer
```

## 🔌 API Usage

### Health Check
```http
GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

### Predict Sentiment
```http
POST http://localhost:8000/predict
Content-Type: application/json

{
  "text": "I love this product! It's amazing!"
}
```

**Response:**
```json
{
  "text": "I love this product! It's amazing!",
  "sentiment": "positive",
  "confidence": 99.87,
  "probabilities": {
    "positive": 99.87,
    "negative": 0.08,
    "neutral": 0.05
  }
}
```

## 📊 Model Performance
- **Algorithm:** Logistic Regression with TF-IDF
- **Training Samples:** 3,600 tweets
- **Test Samples:** 900 tweets
- **Accuracy:** 98.78%
- **Precision:** 99%+ (all classes)
- **Recall:** 99%+ (all classes)

## 🎓 Project Phases Completed

### Phase 1: Model Serialization ✅
- Trained Logistic Regression classifier
- Implemented TF-IDF vectorization
- Saved model and preprocessor using joblib

### Phase 2: Backend API Development ✅
- Built FastAPI application
- Created /health and /predict endpoints
- Implemented Pydantic validation
- Tested with Postman

### Phase 3: Frontend Interface ✅
- Developed Streamlit web interface
- Created interactive input/output components
- Integrated with backend API

### Phase 4: Containerization & Deployment ✅
- Created Dockerfile
- Deployed to Hugging Face Spaces
- Publicly accessible at live URL

## 🎬 Demo Video
[Link to demo video will be added]

## 👨‍💻 Author
**Prajwal Dhoke**
- GitHub: [@PrajwalDhoke](https://github.com/PrajwalDhoke)

## 📄 License
MIT License - Educational Project

## 🙏 Acknowledgments
- End-to-End ML Deployment Course
- Hugging Face for hosting
- FastAPI and Streamlit communities
```
