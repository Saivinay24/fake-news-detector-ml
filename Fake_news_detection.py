

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings


warnings.filterwarnings('ignore')


DATA_FILE = 'WELFake_Dataset.csv'


DEV_MODE = True
DEV_ROWS = 5000


TFIDF_MAX_FEATURES = 5000

TEST_SPLIT_SIZE = 0.2
RANDOM_STATE = 42



def load_data(filepath):
    """
    Loads the WELFake data, cleans it, and reports a summary.
    """
    print(f"--- [Step 1: Loading Data from '{filepath}'] ---")
    
    
    load_rows = DEV_ROWS if DEV_MODE else None
    if DEV_MODE:
        print(f"*** DEV MODE ON: Loading only first {load_rows} rows. ***")
    
    try:
        
        df = pd.read_csv(filepath, usecols=['text', 'label'], nrows=load_rows)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        print("Please download the WELFake dataset and place it in the same directory.")
        return None
    except ValueError:
        print("Error: Could not find required columns 'text' and 'label' in the file.")
        return None

    
    df = df.dropna(subset=['text', 'label'])
    
    
    df['label'] = df['label'].astype(int).map({1: 'real', 0: 'fake'})
    
    
    df = df[df['label'].isin(['fake', 'real'])]

    print(f"Data loaded successfully. Total articles: {len(df)}")
    print("Data summary:")
    print(df.info())
    print("\nLabel distribution:")
    print(df['label'].value_counts())
    print("\n" + "="*50 + "\n")
    return df

def preprocess_and_vectorize(df):
    """
    Performs preprocessing, TF-IDF vectorization, and train/test split.
    """
    print("--- [Step 2: Preprocessing & Feature Engineering] ---")
    
    
    X = df['text']
    y = df['label']

    print(f"Splitting data into {100-TEST_SPLIT_SIZE*100}% train / {TEST_SPLIT_SIZE*100}% test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    
    print(f"Applying TF-IDF vectorizer (max features = {TFIDF_MAX_FEATURES})...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=TFIDF_MAX_FEATURES)
    
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Text data has been vectorized.")
    print(f"Training data shape: {X_train_tfidf.shape}")
    print(f"Test data shape: {X_test_tfidf.shape}")
    print("\n" + "="*50 + "\n")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Trains a model, times it, and prints a full evaluation report.
    """
    print(f"--- [Training Model: {model_name}] ---")
    
    
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    train_time = end_time - start_time
    
    print(f"Training complete in {train_time:.2f} seconds.")

    
    y_pred = model.predict(X_test)
    
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    
    labels = sorted(list(y_test.unique()))
    
    
    plot_confusion_matrix(y_test, y_pred, labels, model_name)
    
    print("\n" + "="*50 + "\n")
    
    return model_name, accuracy, train_time

def plot_confusion_matrix(y_test, y_pred, labels, model_name):
    """
    Generates and saves a confusion matrix plot.
    """
    try:
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        
        
        filename = f"confusion_matrix_{model_name.replace(' ', '_')}.png"
        plt.savefig(filename)
        print(f"Confusion Matrix plot saved to '{filename}'")
        plt.close() 
        
    except Exception as e:
        print(f"Error generating confusion matrix plot: {e}")

def plot_comparison_charts(results_df):
    """
    Generates and saves bar charts comparing model performance.
    """
    print("--- [Step 5: Generating Comparison Charts] ---")
    
    
    results_df = results_df.sort_values(by='Accuracy', ascending=False)
    
    try:
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Accuracy', y='Model', data=results_df, palette='viridis')
        plt.title('Model Accuracy Comparison')
        plt.xlabel('Accuracy')
        plt.xlim(0, 1.0) 
        
        
        for index, row in results_df.iterrows():
            plt.text(row['Accuracy'] + 0.01, index, f"{row['Accuracy']*100:.2f}%", va='center')
            
        acc_filename = "model_accuracy_comparison.png"
        plt.savefig(acc_filename)
        print(f"Accuracy comparison chart saved to '{acc_filename}'")
        plt.close()

       
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Training Time (s)', y='Model', data=results_df, palette='plasma')
        plt.title('Model Training Time Comparison')
        
        time_filename = "model_time_comparison.png"
        plt.savefig(time_filename)
        print(f"Training time comparison chart saved to '{time_filename}'")
        plt.close()

    except Exception as e:
        print(f"Error generating comparison plots: {e}")



def main():
    """
    Main function to run the full pipeline.
    """
    
    df = load_data(DATA_FILE)
    if df is None:
        return 
    
    
    X_train, X_test, y_train, y_test = preprocess_and_vectorize(df)

    
    models_to_test = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    }
    
   
    print("--- [Steps 3 & 4: Training & Evaluating Models] ---")
    
    model_results = []
    
    for model_name, model in models_to_test.items():
        name, accuracy, train_time = train_and_evaluate_model(
            model, X_train, X_test, y_train, y_test, model_name
        )
        model_results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Training Time (s)': train_time
        })
    
    
    if model_results:
        results_df = pd.DataFrame(model_results)
        print("--- [Final Model Comparison Summary] ---")
        print(results_df.to_markdown(index=False))
        print("\n" + "="*50 + "\n")
        
        plot_comparison_charts(results_df)
    
    print("--- Project Pipeline Complete ---")

if __name__ == "__main__":
    main()
