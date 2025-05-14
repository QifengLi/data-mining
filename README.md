# Social Media Sentiment Analysis Project

## Project Introduction
This project implements a machine learning-based social media text sentiment analysis system that can automatically identify sentiment orientation (positive or negative) in text. The project can be applied to brand reputation monitoring, market research, customer feedback analysis, and other fields.

## Features
- Text preprocessing: including lowercase conversion, special character removal, stopword removal, and lemmatization
- Feature extraction: using TF-IDF to vectorize text data
- Model training: implementing logistic regression and random forest classification models
- Model evaluation: including accuracy, precision, recall, F1-score, and other evaluation metrics
- Model application: ability to analyze sentiment in new text data
- Visualization analysis: including data distribution, model performance, and feature importance visualizations

## Installation Dependencies
```bash
pip install -r english_requirements.txt
```

## Usage
1. Prepare the dataset: The dataset should contain 'review' and 'sentiment' columns, where 'sentiment' is 0 for negative sentiment and 1 for positive sentiment
2. Run the main program:
```bash
python sentiment_analysis_english.py
```
3. Custom dataset: Modify the dataset path in the code to use your own dataset

## Project Structure
- `sentiment_analysis_english.py`: Main program file
- `english_requirements.txt`: Project dependencies
- `README_EN.md`: Project documentation
- `sample_sentiment_data.csv`: Sample dataset
- Generated models and charts:
  - `best_sentiment_model.pkl`: Saved best model
  - `sentiment_distribution.png`: Sentiment distribution chart
  - `review_length_distribution.png`: Review length distribution chart
  - `review_length_by_sentiment.png`: Review length by sentiment category chart
  - `logistic_regression_confusion_matrix.png`: Logistic regression model confusion matrix
  - `random_forest_confusion_matrix.png`: Random forest model confusion matrix
  - `feature_importance.png`: Feature importance chart

## Future Extensions
- Add more classification models, such as SVM, LSTM, etc.
- Support multi-language sentiment analysis
- Add sentiment intensity analysis
- Implement a web-based user interface
- Integrate real-time data stream processing functionality 
