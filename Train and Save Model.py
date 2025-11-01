

import pandas as pd
import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')


DATA_FILE = 'WELFake_Dataset.csv'
MODEL_FILE = 'fake_news_model.joblib'
VECTORIZER_FILE = 'fake_news_vectorizer.joblib'


DEV_MODE = False
DEV_ROWS = 5000

TFIDF_MAX_FEATURES = 5000
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42

def train_and_save_model():
    """
    Loads data, trains a model, and saves it to disk.
    """
    print(f"--- [Step 1: Loading Data from '{DATA_FILE}'] ---")
    
    load_rows = DEV_ROWS if DEV_MODE else None
    if DEV_MODE:
        print(f"*** DEV MODE ON: Loading only first {load_rows} rows. ***")
    
    try:
        df = pd.read_csv(DATA_FILE, usecols=['text', 'label'], nrows=load_rows)
    except FileNotFoundError:
        print(f"Error: The file '{DATA_FILE}' was not found.")
        return
    
    df = df.dropna(subset=['text', 'label'])
    df['label'] = df['label'].astype(int).map({1: 'real', 0: 'fake'})
    df = df[df['label'].isin(['fake', 'real'])]
    print(f"Data loaded. Total articles: {len(df)}")
    
    X = df['text']
    y = df['label']

    print(f"--- [Step 2: Splitting Data] ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"--- [Step 3: Vectorizing Text (TF-IDF)] ---")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=TFIDF_MAX_FEATURES)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Text vectorization complete.")
    
    print(f"--- [Step 4: Training Logistic Regression Model] ---")
    start_time = time.time()
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)
    end_time = time.time()
    
    print(f"Model training complete in {end_time - start_time:.2f} seconds.")
    
    
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
    
    print(f"--- [Step 5: Saving Model and Vectorizer] ---")
    
    
    joblib.dump(model, MODEL_FILE)
    print(f"Model saved to: {MODEL_FILE}")
    
    
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print(f"Vectorizer saved to: {VECTORIZER_FILE}")
    
    print("\n--- Model Training and Saving Complete! ---")
    print(f"You can now run the Streamlit app.")

if __name__ == "__main__":
    train_and_save_model()
