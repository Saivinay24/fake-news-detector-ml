

import streamlit as st
import joblib
import os


MODEL_FILE = 'fake_news_model.joblib'
VECTORIZER_FILE = 'fake_news_vectorizer.joblib'


@st.cache_resource
def load_model():
    """
    Loads the model and vectorizer from disk.
    @st.cache_resource ensures this only runs once.
    """
   
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        return None, None
        
    try:
        model = joblib.load(MODEL_FILE)
        vectorizer = joblib.load(VECTORIZER_FILE)
        return model, vectorizer
    except Exception as e:
        print(f"Error loading files: {e}")
        return None, None

model, vectorizer = load_model()


st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("Fake News Detector 🔎")
st.write(
    "Based on the WELFake dataset and a Logistic Regression model. "
    "Paste in a news article's text below to analyze it."
)

if model is None or vectorizer is None:
    st.error(
        "**Error: Model files not found!**\n\n"
        "Please run the `train_and_save_model.py` script first to "
        "generate `fake_news_model.joblib` and `fake_news_vectorizer.joblib`."
    )
else:
    
    
    
    article_text = st.text_area(
        "Paste the full text of the news article here:",
        height=250,
        placeholder="E.g., 'Scientists today announced a groundbreaking discovery...'"
    )
    
    
    if st.button("Analyze News"):
        if not article_text.strip():
            st.warning("Please paste some text to analyze.")
        else:
            
            with st.spinner("Analyzing..."):
                
                text_tfidf = vectorizer.transform([article_text])
                
                
                prediction = model.predict(text_tfidf)[0]
                prediction_proba = model.predict_proba(text_tfidf)[0]
                
                
                confidence = max(prediction_proba) * 100
            
            
            if prediction == 'real':
                st.success(f"**This looks REAL** ({confidence:.2f}% confident)")
                st.balloons()
            else:
                st.error(f"**This looks FAKE** ({confidence:.2f}% confident)")

    st.markdown("---")
    st.caption("Mini-project based on the presentation 'Fake News Detection using NLP and Machine Learning'.")
