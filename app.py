# ================= IMPORTS =================
from pathlib import Path
import streamlit as st
import pickle
import re
import string
import sqlite3
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from newspaper import Article
from deep_translator import GoogleTranslator
from urllib.parse import urlparse
import streamlit.components.v1 as components
import xml.etree.ElementTree as ET

try:
    from lime.lime_text import LimeTextExplainer
    explainability_available = True
except ImportError:
    explainability_available = False

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI News Intelligence Platform",
    page_icon="🧠",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #f0f2f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease 0s;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0px 5px 15px rgba(76, 175, 80, 0.4);
        transform: translateY(-2px);
    }
    .css-1v0mbdj {
        border-radius: 10px;
        padding: 20px;
        background-color: #1e2127;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ================= PATHS =================
script_dir = Path(__file__).parent

model_path = script_dir.parent / "models" / "fake_news_model.pkl"
vectorizer_path = script_dir.parent / "models" / "vectorizer.pkl"
logo_path = script_dir.parent / "images" / "logo.png"

# ================= LOAD MODEL =================
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ================= DATABASE =================
conn = sqlite3.connect("news_ai.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    prediction TEXT,
    confidence REAL,
    time TEXT
)
""")
conn.commit()

# ================= TRUSTED SOURCES =================
trusted_sources = [
    "bbc.com",
    "cnn.com",
    "reuters.com",
    "nytimes.com",
    "theguardian.com",
    "timesofindia.indiatimes.com",
    "indianexpress.com",
    "hindustantimes.com",
    "ndtv.com",
    "aajtak.in",
    "zeenews.india.com",
    "bhaskar.com",
    "amarujala.com",
    "navbharattimes.indiatimes.com",
    "news18.com",
    "lokmat.com",
    "esakal.com",
    "loksatta.com",
    "maharashtratimes.com",
    "abpmajha.abplive.in",
    "tv9marathi.com",
    
    # Additional sources
    "thehindu.com",
    "deccanherald.com",
    "business-standard.com",
    "livemint.com",
    "financialexpress.com",
    "firstpost.com",
    "scroll.in",
    "jansatta.com",
    "patrika.com",
    "livehindustan.com",
    "saamana.com",
    "punyanagari.com",
    "prahaar.in",
    "deshdoot.com",
    "dainikgomantak.com"
]

# ================= CLEAN TEXT =================
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# ================= TRANSLATE =================
def translate_text(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

# ================= SAVE =================
def save(text, pred, prob):
    cursor.execute(
        "INSERT INTO history (text, prediction, confidence, time) VALUES (?, ?, ?, ?)",
        (text, "REAL" if pred == 1 else "FAKE", float(prob), str(datetime.now()))
    )
    conn.commit()

# ================= LOAD HISTORY =================
def load_history():
    return pd.read_sql("SELECT * FROM history ORDER BY id DESC", conn)

# ================= URL EXTRACT =================
def extract_news_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None

# ================= EXPLAINABILITY =================
if explainability_available:
    explainer = LimeTextExplainer(class_names=['Fake', 'Real'])

def explain_prediction(text):
    if not explainability_available:
        return None
    
    def predictor(texts):
        cleaned_texts = [clean_text(t) for t in texts]
        vecs = vectorizer.transform(cleaned_texts)
        return model.predict_proba(vecs)
    
    # We use the text directly so LIME can perturb words
    exp = explainer.explain_instance(text, predictor, num_features=6)
    return exp

# ================= TRUST CHECK =================
def check_source(url):
    try:
        domain = urlparse(url).netloc

        for source in trusted_sources:
            if source in domain:
                return True, domain

        return False, domain

    except:
        return False, "Unknown"

# ================= LIVE NEWS =================
def get_live_news(topic=""):
    try:
        if topic:
            url = f"https://news.google.com/rss/search?q={topic}&hl=en-IN&gl=IN&ceid=IN:en"
        else:
            url = "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en"

        response = requests.get(url)
        root = ET.fromstring(response.content)

        news_list = []
        for item in root.findall('./channel/item')[:5]:
            title = item.find('title').text if item.find('title') is not None else ""
            desc = item.find('description').text if item.find('description') is not None else ""
            clean_desc = re.sub('<.*?>', '', desc) # Remove HTML tags
            news_list.append({"title": title, "description": clean_desc, "text": f"{title} {clean_desc}"})

        return news_list

    except Exception as e:
        return []

# ================= SIDEBAR =================
page = st.sidebar.selectbox(
    "Navigation",
    ["🏠 Home","🧠 Detector","🔗 URL Detector","🌐 Live News","📱 Social Media","📊 Dashboard"]
)

# ================= HOME =================
if page == "🏠 Home":

    col1, col2 = st.columns([1, 4])
    
    with col2:
        st.markdown("<h1 style='font-size: 3rem;'>AI News Intelligence Platform</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.2rem; color: #a0a0a0;'>Empowering you with AI to distinguish facts from fiction in the digital age.</p>", unsafe_allow_html=True)

    st.markdown("---")
    
    st.markdown("### 🌟 Key Features")
    col_feat1, col_feat2, col_feat3 = st.columns(3)
    
    with col_feat1:
        st.markdown("#### 🧠 Smart Detection")
        st.write("Advanced Machine Learning models to instantly classify news articles as Real or Fake with high accuracy.")
        
    with col_feat2:
        st.markdown("#### 🔗 URL & Social Media Analysis")
        st.write("Extract and analyze content directly from news URLs, tweets, and WhatsApp forwards.")
        
    with col_feat3:
        st.markdown("#### 🌐 Live News & Multilingual")
        st.write("Fetch the latest news in real-time. Native support for translating content into English for seamless analysis.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.info("👈 **Get started by selecting an option from the sidebar navigation.**")

# ================= DETECTOR =================
if page == "🧠 Detector":

    st.title("🧠 Fake News Detector")

    with st.form("news_form"):

        text = st.text_area("Paste News Article", height=200)

        col1, col2 = st.columns(2)

        analyze = col1.form_submit_button("🔍 Analyze")
        clear = col2.form_submit_button("🧹 Clear")

    if clear:
        st.rerun()

    if analyze:

        if text.strip() == "":
            st.warning("Enter news text")

        else:

            translated = translate_text(text)
            if translated != text:
                st.info(f"🌐 Translated Text to English: {translated}")

            cleaned = clean_text(translated)

            vec = vectorizer.transform([cleaned])

            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec).max()

            if pred == 0:
                st.error("🚨 Prediction: Fake News")
            else:
                st.success("✅ Prediction: Real News")

            st.progress(float(prob))
            st.write(f"Confidence: {round(prob*100,2)}%")

            if explainability_available:
                with st.expander("🔍 AI Explainability (Why did the model predict this?)"):
                    exp = explain_prediction(translated)
                    if exp:
                        st.write("Important words influencing prediction:")
                        for word, weight in exp.as_list():
                            if weight > 0:
                                st.markdown(f"• **{word}** (Real News Indicator)")
                            else:
                                st.markdown(f"• **{word}** (Fake News Indicator)")
                        components.html(exp.as_html(), height=300, scrolling=True)

            save(text, pred, prob)

# ================= URL DETECTOR =================
if page == "🔗 URL Detector":

    st.title("🔗 URL News Detector")

    url = st.text_input("Paste News URL")

    if st.button("Analyze URL"):

        trusted, domain = check_source(url)

        if trusted:
            st.success(f"Source: {domain}\nStatus: Trusted Source")
        else:
            st.warning(f"Source: {domain}\nStatus: Unknown Source")

        text = extract_news_from_url(url)

        if text:

            translated = translate_text(text)
            cleaned = clean_text(translated)

            vec = vectorizer.transform([cleaned])

            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec).max()

            if pred == 0:
                st.error("🚨 Prediction: FAKE NEWS")
            else:
                st.success("✅ Prediction: REAL NEWS")

            st.progress(float(prob))
            st.write(f"Confidence: {round(prob*100,2)}%")

        else:
            st.error("Unable to extract article text from the provided URL.")

# ================= LIVE NEWS =================
if page == "🌐 Live News":

    st.title("🌐 Live News Analyzer")

    topic = st.text_input("Enter Topic (e.g., Election, Economy, Technology)")

    if st.button("Fetch Live News"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            news_list = get_live_news(topic)

            if not news_list:
                st.error("No news found or API limit reached.")
            
            for i, news in enumerate(news_list):

                st.markdown("---")
                st.write(f"**Article {i+1}:** {news['title']}")
                if news['description']:
                    st.write(f"_{news['description']}_")

                cleaned = clean_text(news['text'])
                vec = vectorizer.transform([cleaned])

                pred = model.predict(vec)[0]
                prob = model.predict_proba(vec).max()

                if pred == 0:
                    st.error("Prediction → FAKE")
                else:
                    st.success("Prediction → REAL")

                st.write(f"Confidence → {round(prob*100,2)}%")

# ================= SOCIAL MEDIA =================
if page == "📱 Social Media":

    st.title("📱 Social Media News Detector")
    st.write("Analyze viral social media posts, tweets, and WhatsApp forwards.")

    with st.form("social_form"):
        social_text = st.text_area("Paste Social Media Text (Twitter, WhatsApp, etc.)", height=150)
        col1, col2 = st.columns(2)
        analyze_social = col1.form_submit_button("🔍 Analyze Post")
        clear_social = col2.form_submit_button("🧹 Clear")
        
    if clear_social:
        st.rerun()
        
    if analyze_social:
        if social_text.strip() == "":
            st.warning("Enter social media text")
        else:
            translated = translate_text(social_text)
            if translated != social_text:
                st.info(f"🌐 Translated to English: {translated}")
                
            cleaned = clean_text(translated)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec).max()

            if pred == 0:
                st.error("🚨 Fake News / Misinformation Detected")
            else:
                st.success("✅ Real / Verified Information")

            st.progress(float(prob))
            st.write(f"Confidence: {round(prob*100,2)}%")

# ================= DASHBOARD =================
if page == "📊 Dashboard":

    st.markdown("<h1 style='text-align: center;'>📊 AI Analytics Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Monitor the performance and statistics of the Fake News Detector.</p>", unsafe_allow_html=True)
    st.markdown("---")

    df = load_history()

    if len(df) > 0:

        # Key Metrics
        col1, col2, col3 = st.columns(3)
        
        total_predictions = len(df)
        avg_confidence = round(df["confidence"].mean()*100, 2)
        real_count = len(df[df["prediction"] == "REAL"])
        fake_count = len(df[df["prediction"] == "FAKE"])
        fake_percentage = round((fake_count / total_predictions) * 100, 1) if total_predictions > 0 else 0

        with col1:
            st.metric("Total Predictions", total_predictions)
        with col2:
            st.metric("Average Confidence", f"{avg_confidence}%")
        with col3:
            st.metric("Fake News Detected", f"{fake_percentage}%", delta=f"{fake_count} articles", delta_color="inverse")

        st.markdown("---")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("### 📈 Prediction Distribution")
            pred_counts = df["prediction"].value_counts().reset_index()
            pred_counts.columns = ["Prediction", "Count"]
            st.bar_chart(pred_counts.set_index("Prediction"))
            
        with col_chart2:
            st.markdown("### 🕒 Recent Confidence Trend")
            recent_conf = df.head(10)[["id", "confidence"]].sort_values("id")
            recent_conf["confidence"] = recent_conf["confidence"] * 100
            st.line_chart(recent_conf.set_index("id"))

        st.markdown("---")
        st.markdown("### 🗄️ Recent Analysis History")
        
        styled_df = df.head(20)[["time", "prediction", "confidence", "text"]].copy()
        styled_df["confidence"] = styled_df["confidence"].apply(lambda x: f"{round(x*100, 2)}%")
        st.dataframe(styled_df, use_container_width=True)

    else:
        st.info("No prediction history available yet. Start analyzing news to see insights!")
        
