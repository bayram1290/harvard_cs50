import os
import sys
import re
import string as strg
from typing import List, Dict, Tuple, Any, Union
from collections import Counter
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import check_array

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('sentiment_analysi'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def loadNLTK():
    """
    Loads necessary NLTK data for sentiment analysis.

    This function downloads the necessary NLTK data for sentiment analysis, including:
        -Punkt tokenizer
        - stopwords
        - wordnet
        - VADER lexicon

    Raises:
        LookupError: If the necessary NLTK data cannot be found.

    Returns:
        None
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        logger.debug('Downloaded NLTK punkt tokenizer')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
        logger.debug('Downloaded NLTK stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
        logger.debug('Downloaded NLTK wordnet')

    try:
        nltk.data.find('sentiment/vader_lexicon')
    except LookupError:
        nltk.download('vader_lexicon')
        logger.debug('Downloaded VADER lexicon')

class SentimentAnalyzer():

    def __init__(self) -> None:
        """
        Initializes the sentiment analyzer.

        This function initializes the sentiment analyzer by loading necessary 
            - NLTK data,
            - creating a WordNet lemmatizer,
            - setting the stop words,
            - creating a VADER sentiment intensity analyzer,
            - creating a TF-IDF vectorizer,
            - setting the classifier and feature words to None.

        Returns:
            None
        """
        loadNLTK()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vader = SentimentIntensityAnalyzer()
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = None
        self.feature_words = None

    def _loadData(self, directory:str) -> Tuple[List[str], List[str]]:
        """
        Loads sentiment labeled data from given directory.

        This function loads the sentiment labeled data from given directory and returns a tuple of lists containing the positive and negative sentences.

        Args:
            directory (str): The path to the directory containing the sentiment labeled data.

        Returns:
            Tuple[List[str], List[str]]: A tuple of lists containing the positive and negative sentences.
        """
        positives, negatives = list(), list()

        for fname, lbl in [("positives.txt", "Positive"), ("negatives.txt", "Negative")]:
            fpath = os.path.join(directory, fname)

            if not os.path.exists(fpath):
                continue

            with open(file=fpath, mode='r', encoding='utf-8') as f:
                lines = f.read().splitlines()

                for line in lines:
                    line = line.strip()
                    if line:
                        if lbl == 'Positive':
                            positives.append(line)
                        elif lbl == 'Negative':
                            negatives.append(line)

        logger.info(f'Total positives: {len(positives)}')
        logger.info(f'Total negatives: {len(negatives)}')

        return positives, negatives

    def _preprocess(self, text: str) -> List[str]:
        """
        Preprocesses a given text by lowering it, removing URLs, @mentions, #hashtags, and punctuation,
        tokenizing it, lemmatizing the tokens, and removing stop words.

        Args:
            text (str): The text to preprocess.

        Returns:
            List[str]: A list of preprocessed tokens.
        """
        text = text.lower()

        # Define patterns to remove URLs, @mentions, #hashtags, and punctuation
        patterns = list([
            # URLs
            r'http\S+',
            # @mentions
            r'@\w+',
            # #hashtags
            r'#\w+',
            # punctuation
            r'[^\w\s.!?]'
        ])

        for pattern in patterns:
            text = re.sub(pattern=pattern, repl='', string=text)

        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # Lemmatize the tokens
        tokens = list([
            # Lemmatize the token
            self.lemmatizer.lemmatize(token)
            # Check if the token is not a stop word and not punctuation
            for token in tokens
            if not token in self.stop_words and not token in strg.punctuation
        ])

        logger.debug(f'Preprocessed tokens: {tokens}')

        # Return the preprocessed tokens
        return tokens

    def _extractFeatures(self, text: str) -> Dict[str, Any]:
        """ Extracts features from a given text.

        This function takes a text string as an input and returns a dictionary containing various features extracted from the text.

        The features extracted include:
        - word count
        - character count
        - average word length
        - exclamation count
        - question count
        - uppercase ratio
        - VADER sentiment scores
        - positive word count
        - negative word count
        - sentiment ratio

        Args:
            text (str): The text to extract features from.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted features.
        """
        tokens = self._preprocess(text)

        features = {
            'word_count': len(tokens), # Count the number of words in the text
            'char_count': len(text), # Count the number of characters in the text
            'avg_word_length': np.mean([len(word) for word in tokens]) if tokens else 0, # Calculate the average word length in the text
            'exclamation_count': text.count('!'), # Count the number of exclamation marks in the text
            'question_count': text.count('?'), # Count the number of question marks in the text
            'uppercase_ratio': sum(1 for char in text if char.isupper()) / len(text) if text else 0, # Calculate the ratio of uppercase characters in the text
        }

        # Extract VADER sentiment scores
        vader_scores = self.vader.polarity_scores(text)
        features.update({
            'vader_compound': vader_scores['compound'], # VADER compound sentiment score
            'vader_positive': vader_scores['pos'], # VADER positive sentiment score
            'vader_negative': vader_scores['neg'], # VADER negative sentiment score
            'vader_neutral': vader_scores['neu'] # VADER neutral sentiment score
        })

        # Define positive and negative words
        positive_words = list(['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like'])
        negative_words = list(['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'worst'])

        # Count the number of positive and negative words in the text
        features['positive_word_count'] = sum(1 for word in tokens if word in positive_words)
        features['negative_word_count'] = sum(1 for word in tokens if word in negative_words)

        # Calculate the sentiment ratio
        features['sentiment_ratio'] = (features['positive_word_count'] - features['negative_word_count']) / len(tokens) if tokens else 0

        if hasattr(self, 'feature_words') and self.feature_words:
            # Iterate through the feature words and check if they are present in the text
            for word in self.feature_words[:100]:
                features[f'has_{word}'] = word in self.feature_words

        logger.debug(f'Extracted features: {len(features)}')

        return features

    def _prepareFeatures(self, textList: List[str], labels=None) -> List:
        """ Prepares features from a list of texts.
        This function takes a list of texts as an input and returns a list of extracted features.
        It first preprocesses each text by tokenizing it, removing stop words, punctuation, and URLs, and lemmatizing the tokens.
        It then counts the frequency of each token and stores the most common 10,000 tokens as feature words.
        Finally, it extracts features from each text using the _extractFeatures method and returns the list of extracted features.

        Args:
            textList (List[str]): A list of texts to prepare features from.
            labels (List[str], optional): A list of labels corresponding to the texts. Defaults to None.

        Returns:
            List: A list of extracted features.
        """
        features, token_list = list(), list()

        # Preprocess each text by tokenizing it, removing stop words, punctuation, and URLs, and lemmatizing the tokens
        for i, txt in enumerate(textList):
            if i % 1000 == 0:
                logger.debug(f'Preprocessing text {i + 1}/{len(textList)}')
            tokens = self._preprocess(txt)
            token_list.extend(tokens)

        # Count the frequency of each token and store the most common 10,000 tokens as feature words
        word_freq = Counter(token_list)
        self.feature_words = [word for word, count in word_freq.most_common(10000)]

        # Extract features from each text
        for i, txt in enumerate(textList):
            if i % 1000 == 0:
                logger.debug(f'Extracting features for text {i + 1}/{len(textList)}')
            features.append(self._extractFeatures(txt))

        logger.info(f'Prepared features for {len(features)} texts')
        return features

    def train(self, directory: str, classifier:str='naive_bayes') -> Union[float, int]:
        """
        Train a sentiment analysis model using the given classifier.

        This function takes a directory path as an input and loads the sentiment labeled data from the directory.
        It then preprocesses the data by tokenizing it, removing stop words, punctuation, and URLs, and lemmatizing the tokens.
        It then extracts features from the preprocessed data and trains a model using the given classifier.

        Args:
            directory (str): The path to the directory containing the sentiment labeled data.
            classifier (str): The type of classifier to use. Defaults to 'naive_bayes'.

        Returns:
            Union[float, int]: The accuracy of the trained model.
        """
        logger.info(f'Training model with "{classifier}" classifier')

        # Load sentiment labeled data from the directory
        positives, negatives = self._loadData(directory)

        if not positives or not negatives:
            err_msg = 'No training data found. Please, check your data files'
            logger.error(err_msg)
            raise ValueError(err_msg)

        text_list = positives + negatives
        label_list = ['Positive'] * len(positives) + ['Negatives'] * len(negatives)

        logger.info(f'Training model on {len(text_list)} total samples')

        # Preprocess the data by tokenizing it, removing stop words, punctuation, and URLs, and lemmatizing the tokens
        features = self._prepareFeatures(text_list)

        # Extract feature vectors from the preprocessed data
        feature_vectors = list()
        for feature_dict in features:
            vector = list()
            for key in sorted(feature_dict.keys()):
                vector.append(feature_dict[key])

            feature_vectors.append(vector)

        # Train the model using the given classifier
        match classifier:
            case 'svm':
                self.classifier = LinearSVC(random_state=42)
                logger.info('Training SVM classifier')
            case 'random_forest':
                self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
                logger.info('Training Random Forest classifier')
            case _:
                self.classifier = MultinomialNB()
                logger.info('Training Naive Bayes classifier')

        self.classifier.fit(feature_vectors, label_list)
        logger.info('Model trained successfully')

        # Evaluate the accuracy of the trained model
        predictions = self.classifier.predict(np.array(feature_vectors))
        accuracy = float(accuracy_score(label_list, predictions))

        logger.info(f'Training completed with accuracy: {(accuracy * 100):.2f} %')
        logger.info(f'Classification Report:\n{classification_report(label_list, predictions)}')

        return accuracy

    def _predict_sentiment(self, input: str):
        """ Predict the sentiment of a given input string.
        This function takes an input string and returns a dictionary containing the sentiment probabilities.
        If the model is not trained yet, it raises a ValueError.
        The function first checks if the model is trained. If it is not, it raises a ValueError.
        It then predicts the sentiment of the input string by extracting features from the string, creating a feature vector from the extracted features, and passing the feature vector to the classifier.
        The classifier then predicts the sentiment probabilities.
        The sentiment probabilities are then processed to only include sentiments with a probability greater than 0.05.

        Args:
            input (str): The input string to predict the sentiment for.

        Returns:
            Dict[str, float]: A dictionary containing the sentiment probabilities.

        Raises:
            ValueError: If the model is not trained yet.
        """
        if self.classifier is None:
            err_msg = 'Model is not trained yet. Please, train the model first'
            logger.error(err_msg)
            raise ValueError(err_msg)

        logger.debug(f"Predicting sentiment for user input: {input[:50]}...")

        # Extract features from the input string
        features = self._extractFeatures(input)
        logger.debug(f"Extracted features: {features}")

        # Create a feature vector from the extracted features
        feature_vector = [features[key] for key in sorted(features.keys())]
        logger.debug(f"Feature vector: {feature_vector}")

        # Get the type of the classifier
        classifier_type = type(self.classifier).__name__

        # Predict the sentiment probabilities
        match classifier_type:
            case 'LinearSVC':
                # The LinearSVC classifier returns the scores of the samples X against the hyperplane of the classifier
                scores = self.classifier.decision_function(check_array([feature_vector]))[0]
                logger.debug(f"Scores: {scores}")

                # Calculate the exponential of the scores
                exp_scores = np.exp(scores)
                logger.debug(f"Exponential scores: {exp_scores}")

                # Calculate the probabilities by dividing the exponential scores by the sum of the exponential scores
                probs = exp_scores / np.sum(exp_scores)
                logger.debug(f"Probabilities: {probs}")

            case 'RandomForestClassifier' | 'MultinomialNB':
                # The Random Forest Classifier and the Multinomial Naive Bayes classifier return the probability estimates for each class
                probs = self.classifier.predict_proba(check_array([feature_vector]))[0]
                logger.debug(f"Probabilities: {probs}")
            case _:
                err_msg = f'Unsupported classifier type: {classifier_type}'
                logger.error(err_msg)
                raise ValueError(err_msg)

        # Get the class names of the classifier
        classes = self.classifier.classes_

        # Create a dictionary containing the sentiment probabilities
        result = {class_name: prob for class_name, prob in zip(classes, probs) if prob > 0.05}
        logger.debug(f"Prediction probabilities: {result}")

        return result

    def analyzeUserInput(self, input:str) -> Dict[str, float]:
        """
        Analyzes the sentiment for the given user input.

        Args:
            input (str): The input string to analyze the sentiment for.

        Returns:
            Dict[str, float]: A dictionary containing the sentiment probabilities.

        Raises:
            ValueError: If the model is not trained yet.
        """
        logger.info(f'Analyzing sentiment for user input:\n\n"{input}"\n')

        # Predict the sentiment probabilities using the classifier
        probs = self._predict_sentiment(input)

        # Get the sentiment scores using VADER
        vader_scores = self.vader.polarity_scores(input)

        logger.info('\n' + '=' * 50 + '\nSentiment Analysis Results\n' + '=' * 50)
        logger.info(f'Text: {input}\nClassifier Probabilities:')
        for sentiment, prob in probs.items():
            logger.info(f'  {sentiment}: {prob: .4f}')

        logger.info('VADER Sentiment Scores:')
        for key, score in vader_scores.items():
            logger.info(f'  {key}: {score:.4f}')

        # Get the sentiment with the highest probability
        max_sentiment = max(probs.items(), key=lambda x: x[1])
        logger.info(f'Most likely sentiment: {max_sentiment[0]} ({(max_sentiment[1] * 100):.2f}% confidence)')

        return probs

def main():
    """ Main function for the sentiment analysis app.

    This function will train the sentiment analyzer model if the corpus directory is provided.
    Otherwise, it will start an interactive sentiment analysis session.

    :raises Exception: If the training fails.
    :raises KeyboardInterrupt: If the user presses Ctrl+C to exit the program.
    :raises Exception: If an error occurs during sentiment analysis.
    """
    if len(sys.argv) != 2:
        err_msg = 'App usage: python app.py path_to_corpus_directory'
        logger.error(err_msg)
        sys.exit(err_msg)

    analyzer = SentimentAnalyzer()

    try:
        accuracy = analyzer.train(sys.argv[1], classifier='random_forest')
        logger.info(f'Training completed with accuracy: {(accuracy * 100):.2f} %')
    except Exception as e:
        logger.error(f'Training failed: {str(e)}')
        sys.exit(1)

    logger.info('Starting interactive sentiment analysis')

    running = True
    while running:
        try:
            user_input = str(input('\nEnter text to analyze (or "quit" to exit):  ')).strip()

            if user_input.lower() == 'quit':
                logger.info('\nQuiting the program ...')
                running = False

            if not user_input:
                logger.warning('Empty input recieved')
                continue

            analyzer.analyzeUserInput(user_input)

        except KeyboardInterrupt:
            logger.info('\nExiting the program ...')
            running = False
        except Exception as e:
            logger.error(f'Error during analysis: {str(e)}')
            continue

if __name__ == '__main__':
    main()
