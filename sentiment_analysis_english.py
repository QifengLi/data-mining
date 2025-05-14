#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import time
import pickle

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class SentimentAnalysisModel:
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def preprocess_text(self, text):
        """Text preprocessing function, including lowercase conversion, special character removal, stopword removal, and lemmatization"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(cleaned_tokens)
    
    def load_data(self, filepath):
        """Load dataset"""
        print(f"Loading dataset: {filepath}")
        df = pd.read_csv(filepath)
        print(f"Dataset loaded successfully, total records: {len(df)}")
        return df
    
    def explore_data(self, df):
        """Data exploration"""
        print("\nData Exploration:")
        print(f"Data dimensions: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        print("\nBasic statistics:")
        print(df.describe())
        print("\nTarget variable distribution:")
        print(df['sentiment'].value_counts())
        
        # Visualize sentiment distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x='sentiment', data=df)
        plt.title('Sentiment Distribution')
        plt.savefig('sentiment_distribution.png')
        
        # Review length analysis
        df['review_length'] = df['review'].apply(len)
        plt.figure(figsize=(10, 6))
        sns.histplot(df['review_length'], bins=50)
        plt.title('Review Length Distribution')
        plt.xlabel('Review Length')
        plt.savefig('review_length_distribution.png')
        
        # View review length by sentiment category
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='sentiment', y='review_length', data=df)
        plt.title('Review Length by Sentiment Category')
        plt.savefig('review_length_by_sentiment.png')
        
        return df
    
    def prepare_data(self, df):
        """Data preparation"""
        print("\nPreparing data...")
        # Apply text preprocessing
        df['processed_review'] = df['review'].apply(self.preprocess_text)
        print("Text preprocessing completed")
        
        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_review'], 
            df['sentiment'], 
            test_size=0.2, 
            random_state=42
        )
        
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def build_and_train_models(self, X_train, y_train):
        """Build and train models"""
        print("\nStarting model training...")
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000)
        
        # Logistic Regression model
        lr_start_time = time.time()
        lr_pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ])
        lr_pipeline.fit(X_train, y_train)
        lr_train_time = time.time() - lr_start_time
        print(f"Logistic Regression model training completed, time taken: {lr_train_time:.2f} seconds")
        
        # Random Forest model
        rf_start_time = time.time()
        rf_pipeline = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        rf_pipeline.fit(X_train, y_train)
        rf_train_time = time.time() - rf_start_time
        print(f"Random Forest model training completed, time taken: {rf_train_time:.2f} seconds")
        
        return {
            'logistic_regression': lr_pipeline,
            'random_forest': rf_pipeline
        }
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate models"""
        print("\nModel Evaluation:")
        
        results = {}
        for name, model in models.items():
            print(f"\nEvaluating {name} model:")
            # Prediction
            y_pred = model.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.4f}")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Negative', 'Positive'], 
                        yticklabels=['Negative', 'Positive'])
            plt.title(f'{name} Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(f'{name}_confusion_matrix.png')
            
            results[name] = {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'confusion_matrix': cm
            }
        
        return results
    
    def save_best_model(self, models, results):
        """Save the best model"""
        # Select the model with the highest accuracy
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_model = models[best_model_name]
        
        print(f"\nThe best model is {best_model_name}, with an accuracy of: {results[best_model_name]['accuracy']:.4f}")
        
        # Save the model
        with open('best_sentiment_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        print("Best model saved as 'best_sentiment_model.pkl'")
        
        return best_model_name, best_model
    
    def feature_importance(self, models):
        """Feature importance analysis"""
        if 'random_forest' in models:
            rf_model = models['random_forest']
            # Get feature names
            feature_names = rf_model.named_steps['vectorizer'].get_feature_names_out()
            # Get feature importances
            importances = rf_model.named_steps['classifier'].feature_importances_
            
            # Create feature importance DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            
            # Sort by importance and take the top 20
            top_features = feature_importance_df.sort_values('importance', ascending=False).head(20)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='importance', y='feature', data=top_features)
            plt.title('Top 20 Important Features in Random Forest Model')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            
            print("\nFeature importance analysis completed, top 10 most important features:")
            print(top_features.head(10))
    
    def predict_sentiment(self, model, text):
        """Use the model to predict sentiment of new text"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        # Predict
        sentiment = model.predict([processed_text])[0]
        probability = model.predict_proba([processed_text])[0]
        
        result = {
            'text': text,
            'sentiment': sentiment,
            'confidence': max(probability)
        }
        
        return result
    
    def run_pipeline(self, data_path):
        """Run the complete data mining pipeline"""
        # 1. Load data
        df = self.load_data(data_path)
        
        # 2. Data exploration
        df = self.explore_data(df)
        
        # 3. Data preparation
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # 4. Build and train models
        models = self.build_and_train_models(X_train, y_train)
        
        # 5. Evaluate models
        results = self.evaluate_models(models, X_test, y_test)
        
        # 6. Save best model
        best_model_name, best_model = self.save_best_model(models, results)
        
        # 7. Feature importance analysis
        self.feature_importance(models)
        
        # 8. Example predictions
        sample_texts = [
            "This product is amazing, I love its design and functionality!",
            "Poor quality, it broke after one week, waste of money",
            "It's okay, but a bit expensive",
            "Customer service was excellent, they solved my problem"
        ]
        
        print("\nExample Predictions:")
        for text in sample_texts:
            result = self.predict_sentiment(best_model, text)
            sentiment_label = "Positive" if result['sentiment'] == 1 else "Negative"
            print(f"Text: {result['text']}")
            print(f"Predicted sentiment: {sentiment_label}, Confidence: {result['confidence']:.4f}")
            print("-" * 50)
        
        return {
            'df': df,
            'models': models,
            'results': results,
            'best_model_name': best_model_name,
            'best_model': best_model
        }

# Main function
if __name__ == "__main__":
    # Create sentiment analysis model
    sentiment_model = SentimentAnalysisModel()
    
    # Assume we have a sentiment analysis dataset
    # For actual use, please replace the path with your dataset path
    # The dataset should contain 'review' and 'sentiment' columns
    # For demonstration, we'll create a small sample dataset
    
    # Create sample dataset
    sample_data = {
        'review': [
            "I love this product, it's amazing!",
            "Terrible quality, broke after a week.",
            "Good value for money, would recommend.",
            "Not worth the price, very disappointed.",
            "Customer service was excellent.",
            "The product is okay but not great.",
            "Best purchase I've made this year!",
            "Waste of money, don't buy it.",
            "Average product, nothing special.",
            "Very happy with my purchase."
        ],
        'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 0, 1]  # 1=positive, 0=negative
    }
    
    sample_df = pd.DataFrame(sample_data)
    sample_df.to_csv('sample_sentiment_data.csv', index=False)
    
    # Run the complete data mining pipeline
    results = sentiment_model.run_pipeline('sample_sentiment_data.csv') 