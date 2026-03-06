# 🐦 Twitter Sentiment Analyzer - End-to-End ML Project

## 🌐 Live Demo
**Try it here:** https://huggingface.co/spaces/PrajwalDhoke/twitter_sentiment_analyzer

## 📊 Project Overview
A complete end-to-end machine learning application that analyzes sentiment in text using Natural Language Processing. Built as a capstone project demonstrating the full ML deployment pipeline.

## 🎯 Features
- **Sentiment Analysis:** Classifies text as Positive, Negative, or Neutral
- **High Accuracy:** 98%+ accuracy on sample dataset
- **REST API:** FastAPI backend with automatic documentation
- **Web Interface:** Interactive Streamlit frontend
- **Dockerized:** Fully containerized application
- **Cloud Deployed:** Hosted on Hugging Face Spaces

## 🛠️ Tech Stack
- **ML:** scikit-learn, Logistic Regression, TF-IDF
- **Backend:** FastAPI, Uvicorn, Pydantic
- **Frontend:** Streamlit
- **Deployment:** Docker, Hugging Face Spaces
- **Version Control:** Git, GitHub

## 🚀 Local Installation

### Prerequisites
- Python 3.12+
- pip
- Virtual environment (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/PrajwalDhoke/twitter-sentiment-analyzer.git
cd twitter-sentiment-analyzer

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Train model (if needed)
python train_model.py
```

## 🏃 Running Locally

### Option 1: Run API + Frontend Separately

**Terminal 1 - API:**
```bash
python run_api.py
```
API will be available at: http://localhost:8000

**Terminal 2 - Frontend:**
```bash
python run_frontend.py
```
Frontend will be available at: http://localhost:8501

### Option 2: Run with Docker
```bash
# Build image
docker build -t sentiment-analyzer .

# Run container
docker run -p 8000:8000 -p 8501:8501 sentiment-analyzer
```

## 🔌 API Usage

### Health Check
```bash
GET http://localhost:8000/health
```

### Predict Sentiment
```bash
POST http://localhost:8000/predict
Content-Type: application/json

{
  "text": "I love this product!"
}
```

**Response:**
```json
{
  "text": "I love this product!",
  "sentiment": "positive",
  "confidence": 99.87,
  "probabilities": {
    "positive": 99.87,
    "negative": 0.08,
    "neutral": 0.05
  }
}
```

## 📁 Project Structure
```
twitter-sentiment-analyzer/
├── model/              # Trained model files
├── app/                # FastAPI backend
├── frontend/           # Streamlit UI
├── data/               # Training dataset
├── Dockerfile          # Docker configuration
└── requirements.txt    # Python dependencies
```

## 📈 Model Performance
- **Algorithm:** Logistic Regression
- **Feature Engineering:** TF-IDF Vectorization
- **Accuracy:** 92.78%
- **Training Samples:** 3,600
- **Test Samples:** 900

## 🎓 Project Phases
1. ✅ **Model Development** - Built and trained sentiment classifier
2. ✅ **Backend API** - Created FastAPI endpoints
3. ✅ **Frontend UI** - Developed Streamlit interface
4. ✅ **Containerization** - Dockerized application
5. ✅ **Cloud Deployment** - Deployed to Hugging Face Spaces

## 📹 Demo Video
[Link to demo video - will be added]

## 👨‍💻 Author
**Prajwal Dhoke**
- GitHub: [@PrajwalDhoke](https://github.com/PrajwalDhoke)
- Project: End-to-End ML Capstone

## 📄 License
MIT License - Educational Project

## 🙏 Acknowledgments
- Dataset: Sample sentiment data
- Framework: FastAPI, Streamlit
- Deployment: Hugging Face Spaces
```
- Phase 3: Created an interactive Streamlit frontend
- Phase 4: Containerized with Docker and deployed to the cloud"

[09:00 - Conclusion]
"The model achieves 98% accuracy and is now publicly accessible
at the URL shown. Thank you!"
