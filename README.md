# 🧠 AI News Intelligence Platform

An AI-powered Fake News Detection system that analyzes news articles, URLs, and social media posts to determine whether the information is **Real or Fake** using Machine Learning.

---

## 🚀 Project Overview

The **AI News Intelligence Platform** is designed to help users detect misinformation circulating on the internet.
The system uses Natural Language Processing (NLP) and Machine Learning to analyze text and classify news content.

The platform can analyze:

* News articles
* News URLs
* Social media posts (Twitter, WhatsApp, etc.)
* Live news topics

It also includes an **AI Explainability feature** that shows why the model predicted the result.

---

## ✨ Features

### 🧠 Fake News Detection

Classifies news text into:

* **Real News**
* **Fake News**

with confidence score.

### 🔗 URL News Analyzer

Extracts article content directly from a news URL and analyzes it.

### 📱 Social Media Post Detection

Detects misinformation from viral posts, tweets, and WhatsApp forwards.

### 🌐 Live News Analyzer

Fetches live news using RSS feeds and analyzes them automatically.

### 🔍 Explainable AI (XAI)

Uses LIME to show the **important words influencing the model's prediction**.

### 🌍 Multilingual Support

Automatically translates news into English for analysis.

### 📊 AI Analytics Dashboard

Shows:

* Prediction statistics
* Confidence trends
* Analysis history

---

## 🛠 Tech Stack

**Frontend**

* Streamlit

**Backend**

* Python

**Machine Learning**

* Scikit-learn
* NLP Text Vectorization

**Explainable AI**

* LIME

**Other Libraries**

* Pandas
* Matplotlib
* Newspaper3k
* Deep Translator
* SQLite

---

## 📂 Project Structure

```
Fake-News-Detection-AI
│
├── app.py
├── requirements.txt
├── packages.txt
│
├── models
│   ├── fake_news_model.pkl
│   └── vectorizer.pkl
│
├── images
│   └── logo.png
│
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/your-username/Fake-News-Detection-AI.git
```

Go to the project folder:

```
cd Fake-News-Detection-AI
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit app:

```
streamlit run app.py
```

The application will open in your browser.

---

## 📊 How It Works

1. User enters news text or URL.
2. Text is cleaned using NLP preprocessing.
3. Text is converted to numerical vectors.
4. Machine Learning model predicts **Real or Fake**.
5. Confidence score and explanation are displayed.

---

## 🔮 Future Improvements

* Deep Learning model (BERT)
* Voice-based fake news detection
* Image-based misinformation detection
* Real-time fact-check API integration
* Browser extension for fake news detection

---

## 👩‍💻 Author

Developed by **Vaishnavi Bagal**

Computer Science Engineering Student
AI & Data Science Enthusiast

---

## 📜 License

This project is for **educational and research purposes**.
