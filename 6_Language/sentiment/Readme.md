# Sentiment Analyzer

A sophisticated Python-based sentiment analysis tool that combines multiple NLP techniques and machine learning classifiers to determine text sentiment with high accuracy.

## Features

- **Multiple Classifiers**: Naive Bayes, SVM, and Random Forest
- **Advanced Preprocessing**: Lemmatization, stopword removal, URL cleaning
- **Hybrid Analysis**: Combines custom classifier with VADER sentiment scores
- **Rich Feature Extraction**: TF-IDF, emotional words, text statistics, punctuation analysis
- **Detailed Reporting**: Confidence scores, comparative analysis, training metrics

## Installation

```bash
pip install nltk scikit-learn
```

## Usage

### Training

```bash
python app.py /path/to/corpus
```

### Input Files

- `positives.txt` - Positive training samples
- `negatives.txt` - Negative training samples

### Interactive Analysis

The program provides an interactive interface where you can input text and receive detailed sentiment analysis with probabilities.

## Output

The analyzer provides:

- Classifier probabilities for Positive/Negative
- VADER sentiment scores (compound, positive, negative, neutral)
- Overall sentiment determination with confidence percentage
- Training accuracy and classification reports

## Algorithm

1. **Text Preprocessing**: Tokenization, lemmatization, stopword removal
2. **Feature Extraction**: Statistical features, emotional words, VADER scores
3. **Model Training**: Multiple classifier options with cross-validation
4. **Sentiment Prediction**: Combined probability analysis

## Supported Classifiers

- `naive_bayes` - Traditional Naive Bayes
- `svm` - Support Vector Machine
- `random_forest` - Ensemble Random Forest (default)

## Accuracy

The system typically achieves high accuracy through feature engineering and hybrid analysis approaches.\*\*\*\*
