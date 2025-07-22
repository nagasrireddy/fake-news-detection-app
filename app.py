import streamlit as st
import pickle
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import shap
import scipy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from bert_model import predict_bert  # Import from your new file

# List of potentially unreliable sources
untrusted_sources = ["whatsapp", "facebook", "youtube", "blogspot", "reddit", "tiktok", "telegram", "snapchat"]

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# Sidebar
st.sidebar.title("📊 Model Information")
model_choice = st.sidebar.selectbox("Select a model:", ["Logistic Regression", "Naive Bayes", "BERT (DistilBERT)"])
st.sidebar.markdown("**Dataset:** Kaggle Fake & Real News")
st.sidebar.markdown("Built by [Korukanti Nagasri Reddy](https://github.com/nagasrireddy)")

# Load models (for traditional ML)
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

if model_choice == "Logistic Regression":
    model = pickle.load(open('model/lr_model.pkl', 'rb'))
    st.sidebar.markdown("**Model:** Logistic Regression")
    st.sidebar.markdown("**Accuracy:** ~98.5%")
elif model_choice == "Naive Bayes":
    model = pickle.load(open('model/nb_model.pkl', 'rb'))
    st.sidebar.markdown("**Model:** Naive Bayes")
    st.sidebar.markdown("**Accuracy:** ~96.7%")
else:
    st.sidebar.markdown("**Model:** DistilBERT (Hugging Face Transformers)")
    st.sidebar.markdown("**Accuracy:** ~90%+ (context-aware)")

# Load dataset for wordcloud
@st.cache_data
def load_data():
    fake = pd.read_csv("dataset/Fake.csv")
    true = pd.read_csv("dataset/True.csv")
    return fake, true

fake_df, true_df = load_data()

# Word cloud plotter
def plot_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=20)
    ax.axis("off")
    st.pyplot(fig)

# Main title
st.title("📰 Fake News Detection App")
st.write("Check if a news article or headline is **real or fake** using Machine Learning.")

# User input
user_input = st.text_area("📝 Enter news article or headline below:", height=200)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    elif len(user_input.split()) < 8:
        st.warning("⚠️ Please enter a longer news article (8+ words) for accurate prediction.")
    else:
        # 🚨 Source trust checker
        for src in untrusted_sources:
            if src in user_input.lower():
                st.warning(f"⚠️ The text mentions '{src}'. This may be an unreliable or unverified source.")
                break

        # 🤖 BERT Prediction
        if model_choice == "BERT (DistilBERT)":
            pred, label, confidence = predict_bert(user_input)
            if pred == 1:
                st.markdown("### ✅ Prediction: **REAL NEWS**")
                st.success(f"👍 This news appears to be **reliable**. (Confidence: {confidence:.2f})")
            else:
                st.markdown("### ❌ Prediction: **FAKE NEWS**")
                st.error(f"🚨 This appears to be **misleading or false**. (Confidence: {confidence:.2f})")

        # 📊 Traditional ML (LogReg or NB)
        else:
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)

            if prediction[0] == 0:
                st.markdown("### ❌ Prediction: **FAKE NEWS**")
                st.error("🚨 Be cautious. This news appears to be **false or misleading**.")
            else:
                st.markdown("### ✅ Prediction: **REAL NEWS**")
                st.success("👍 This news appears to be **reliable and trustworthy**.")

            # SHAP Explainability (only for Logistic Regression)
            if model_choice == "Logistic Regression":
                st.markdown("### 🧠 Top Words That Influenced the Prediction")

                explainer = shap.LinearExplainer(
                    model,
                    vectorizer.transform(model.classes_.astype(str)),
                    feature_perturbation="interventional"
                )
                shap_values = explainer(input_vec)

                words = vectorizer.build_analyzer()(user_input)
                dense_vec = input_vec.toarray()[0]
                feature_names = vectorizer.get_feature_names_out()

                word_scores = {}
                for word in words:
                    if word in feature_names:
                        index = list(feature_names).index(word)
                        word_scores[word] = shap_values.values[0][index]

                top_words = sorted(word_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                for word, score in top_words:
                    st.write(f"• **{word}** → {score:.4f}")
            elif model_choice == "Naive Bayes":
                st.info("🧠 SHAP explainability not available for Naive Bayes.")

        # 💬 Sentiment Analysis
        st.markdown("### 💬 Sentiment Analysis")
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(user_input)
        compound = scores['compound']

        if compound >= 0.05:
            sentiment = "Positive 😊"
        elif compound <= -0.05:
            sentiment = "Negative 😡"
        else:
            sentiment = "Neutral 😐"

        st.info(f"**Sentiment:** {sentiment}")

# Word clouds
st.markdown("---")
st.subheader("📊 Word Cloud Visualizations")

col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🔴 Fake News Keywords")
    plot_wordcloud(fake_df['text'].astype(str), "Fake News")
with col2:
    st.markdown("### 🟢 Real News Keywords")
    plot_wordcloud(true_df['text'].astype(str), "Real News")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by [Korukanti Nagasri Reddy](https://github.com/nagasrireddy)")
