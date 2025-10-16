# Fake-News-Detector
Fake News Detection using Machine Learning (Streamlit Web App)
🔍 Overview

This project is an AI-powered Fake News Detection System that uses Natural Language Processing (NLP) and Machine Learning to classify news articles as real or fake.
It features a Streamlit web application for interactive predictions and insights.

🚀 Key Features

✅ Real-time Fake News Prediction — Paste any headline or article to get instant results.
✅ Explainable AI — Visualizes top contributing keywords and n-grams behind each prediction.
✅ Batch Upload Support — Upload CSV files for bulk predictions and download results.
✅ Deploy-Ready — Easily hostable on Streamlit Community Cloud or locally.
✅ Lightweight Model — TF-IDF + Logistic Regression pipeline for fast and accurate classification.

🧠 Tech Stack

Language: Python

Frameworks: Streamlit, Scikit-learn, Pandas, NumPy

Model: TF-IDF Vectorizer + Logistic Regression

Deployment: Streamlit Cloud / Localhost

Dataset: Fake and Real News Dataset from Kaggle

📈 Model Performance
Metric	Score
Accuracy	~98%
Precision	~97%
Recall	~98%
F1 Score	~97%
ROC AUC	~0.98

(Exact metrics depend on train/test splits)

⚙️ How to Run Locally
git clone https://github.com/JAYPRAKASH-ROY/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
python train.py     # trains and saves model.pkl
streamlit run streamlit_app.py


Then open the URL shown in your terminal (usually http://localhost:8501
).

☁️ Deployment on Streamlit Cloud

Push your repo to GitHub.

Go to https://share.streamlit.io
.

Connect your GitHub repo and set Main file path → streamlit_app.py.

Deploy! 🚀

📦 Files Included
File	Description
True.csv, Fake.csv	Dataset files
train.py	Training script
model.pkl	Saved ML model
streamlit_app.py	Streamlit web app
requirements.txt	Dependencies
metrics.json	Model evaluation metrics
submission.csv	Sample predictions
👨‍💻 Author

Jayprakash Roy
🎓 B.Sc (Hons) Data Science — Amity University, Kolkata
💡 Interests: Machine Learning, Data Science, and Generative AI
📫 GitHub: JAYPRAKASH-ROY
