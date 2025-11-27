import sys
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter
from pathlib import Path
import re
import json

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import argparse

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s-%(levelnam)s-%(message)s',
    handlers=[logging.FileHandler('ngram_analyzer.log'), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def obtainNLTK():
    """
    This function will download the required NLTK libraries if they are not available.

    The required libraries are:
    - 'punkt': for tokenization
    - 'stopwords': for removing stopwords
    - 'wordnet': for lemmatization
    - 'avg_perception_tagger': for part-of-speech tagging
    """
    required_libs = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'avg_perception_tagger': 'taggers/averaged_perceptron_tagger'
    }

    for data_name, data_path in required_libs.items():
        try:
            # Check if the data is already available
            nltk.data.find(data_path)
            logger.debug(f'NLTK {data_name} already available')

            # If the 'punkt' library is available, try to download 'punkt_tab'
            if data_name == 'punkt':
                try:
                    nltk.download('punkt_tab', quiet=True)
                except:
                    logger.warning('Could not download punkt_tab')
        except LookupError:
            # If the data is not available, download it
            logger.info(f'Downloading NLTK {data_name} ...')
            nltk.download(data_name, quiet=True)
            logger.info(f'Downloaded NLTK: {data_name}')
        except Exception as e:
            # If there is an error while downloading, log a warning
            logger.warning(f'Failed to download {data_name}')

@dataclass
class AnalyzerConfig():
    n: int=2
    top_k: int=20
    min_freq: int=2
    use_stopwords: bool=True
    use_stem: bool=False
    use_lemmatize: bool=False
    case_sensative: bool=False
    include_punctuation: bool=False
    detect_characters: bool=True
    assess_dialogue_patterns: bool=True
    min_file_size: int=100
    output_format: str='text'
    save_results: bool=False

class RobustTokenizer():
    def __init__(self) -> None:
        self._setupTokenizers()

    def _setupTokenizers(self) -> None:
        """
        This function is used to set up the tokenization system.
        It will first try to use the NLTK library's tokenizers.
        If NLTK is not available, it will default to using regex.
        """
        self.use_nltk = False
        self.nltk_available = False

        # Try to find the NLTK punkt tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
            self.use_nltk = True
            self.nltk_available = True
            logger.info('Using NLTK tokenizers')
        except LookupError:
            # If NLTK is not available, use regex tokenizers
            logger.warning('NLTK punkt not available, using regex tokenizers')

    def tokenizeSentence(self, content: str) -> List[str]:
        """
        Tokenize a string of content into individual sentences.

        If the NLTK library is available, use it to tokenize the sentences.
        Otherwise, use a regex to split the content into sentences.
        """
        if self.use_nltk:
            try:
                # Use NLTK to tokenize the sentences
                return nltk.sent_tokenize(content)
            except Exception as e:
                # If NLTK fails, fall back to using regex
                self.use_nltk = False
                logger.warning(f'NLTK sentence tokenization failed: {e}')

        # Split the content into sentences using a regex
        sentences = re.split(r'[.?!]+[\s\n]+', content)
        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def wordTokenize(self, txt: str) -> List[str]:
        """
        Tokenize a string of text into individual words.

        If the NLTK library is available, use it to tokenize the words.
        Otherwise, use a regex to split the content into words.

        The regex used is \b[\w\']+\b[.,!?;:]\'. This will match any word
        that is followed by a punctuation mark.

        The words are then stripped of any leading or trailing whitespace.
        """
        if self.use_nltk:
            try:
                # Use NLTK to tokenize the words
                return nltk.word_tokenize(txt)
            except Exception as e:
                # If NLTK fails, fall back to using regex
                self.use_nltk = False
                logger.warning(f'NLTK word tokenization failed: {e}')

        # Split the content into words using a regex
        words = re.findall(r'\b[\w\']+\b[.,!?;:]', txt)
        return [word.strip() for word in words if word.strip()]

class TextProcessor():

    def __init__(self, config: AnalyzerConfig) -> None:
        """
        Initialize the TextProcessor object.

        This method takes an AnalyzerConfig object as a parameter and
        initializes the TextProcessor object with the given configuration.

        It sets up the tokenizer, stemmer, lemmatizer, and stop words
        based on the configuration.

        The stop words are a set of common English words that are ignored
        during the analysis. The dialogue verbs are a set of verbs that are
        commonly used in dialogue such as "said", "asked", etc. The
        character indicators are a set of words that are commonly used to
        indicate a character such as "mr", "mrs", etc.
        """

        self.config = config
        self.tokenizer = RobustTokenizer()
        self.stemmer = PorterStemmer() if self.config.use_stem else None
        self.lemmatizer = WordNetLemmatizer() if self.config.use_lemmatize else None

        self.stop_words = set()
        if config.use_stopwords:
            try:
                # Load the list of stop words from NLTK
                self.stop_words = set(stopwords.words('english'))
            except Exception as e:
                logger.warning(f'Could not load sentences: {e}')
                self.stop_words = {
                    # List of common English stop words
                    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", 
                    "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 
                    'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                    'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 
                    'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
                    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
                    'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
                    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
                    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
                    'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 
                    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 
                    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 
                    'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 
                    'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 
                    'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', 
                    "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
                    'wouldn', "wouldn't"
                }

            self.dialogue_verbs = {
                # List of verbs commonly used in dialogue
                'said', 'asked', 'replied', 'cried', 'exclaimed', 'shouted', 
                'whispered', 'muttered', 'answered', 'continued', 'added',
                'observed', 'remarked', 'declared', 'announced', 'stated'
            }

            self.character_indicators = {
                # List of words commonly used to indicate a character
                'mr', 'mrs', 'miss', 'dr', 'professor', 'sir', 'lord', 'lady',
                'captain', 'colonel', 'inspector', 'detective', 'officer'
            }

    def _sanitizeTxt(self, content: str) -> str:
        """
        Sanitizes a given text by replacing multiple spaces with a single space,
        replacing multiple newline characters with a single newline character, and adding a period
        at the end of each line if it does not already end with a sentence-ending punctuation mark.

        The function first checks if the file exists. If not, it raises a FileNotFoundError.
        It then attempts to read the file with different encodings (utf-8, latin-1, and cp1252) until it succeeds.
        If it fails with all encodings, it raises a ValueError.

        Once the text is loaded, it preprocesses the text by replacing multiple spaces with a single space,
        replacing multiple newline characters with a single newline character, and adding a period at the end of each line
        if it does not already end with a sentence-ending punctuation mark.

        Args:
            content (str): The text to preprocess.

        Returns:
            str: The preprocessed text.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file could not be decoded with any common encoding.
        """

        # Replace multiple spaces with a single space
        text = re.sub(pattern=r'\s+', repl=' ', string=content)

        # Replace multiple newline characters with a single newline character
        text = re.sub(pattern=r'\n\s*\n', repl='\n\n', string=text)

        # Remove any Project Gutenberg header/footer text
        text = re.sub(pattern=r'\*+\s*START[^*]+\*+', repl='', string=text, flags=re.IGNORECASE)
        text = re.sub(pattern=r'\*+\s*END[^*]+\*+', repl='', string=text, flags=re.IGNORECASE)

        # Remove any references to Project Gutenberg
        text = re.sub(pattern=r'project gutenberg', repl='', string=text, flags=re.IGNORECASE)

        # Remove any page number references
        text = re.sub(pattern=r'\bPage\s+\d+\b', repl='', string=text, flags=re.IGNORECASE)

        # Remove any lines that only contain a single number
        text = re.sub(pattern=r'^\d+$', repl='', string=text, flags=re.MULTILINE)

        # Strip leading and trailing whitespace from the text
        return text.strip()

    def preprocessTxt(self, content: str) -> List[str]:
        """
        This method takes a given text and preprocesses it by removing punctuation, stop words, and stemming/lemmatizing the words.

        It first sanitizes the text by replacing multiple spaces with a single space, replacing multiple newline characters with a single newline character, and adding a period at the end of each line if it does not already end with a sentence-ending punctuation mark.

        It then tokenizes the text into individual words using the RobustTokenizer.

        It then loops through each token and checks if it meets the following conditions:
            - If the token is empty, it skips it.
            - If the configuration is set to be case sensitive, it converts the token to lowercase.
            - If the configuration is set to not use stop words and the token is a stop word, it skips it.
            - If the configuration is set to not include punctuation and the token is a punctuation mark, it skips it.
            - If the token is a number and its length is less than 5, it skips it.

        If the configuration is set to use stemming or lemmatization, it applies the stemming or lemmatization to the token.

        Finally, it adds the processed token to the list of processed tokens and returns the list.

        Args:
            content (str): The text to preprocess.

        Returns:
            List[str]: The list of processed tokens.
        """

        sanitized_txt = self._sanitizeTxt(content)
        tokens = self.tokenizer.wordTokenize(txt=sanitized_txt)

        processed_tokens = list()

        for token in tokens:
            # If the token is empty, skip it
            if not token.strip():
                continue

            # If the configuration is set to be case sensitive, convert the token to lowercase
            if self.config.case_sensative:
                token = token.lower()

            # If the configuration is set to not use stop words and the token is a stop word, skip it
            if not self.config.use_stopwords and token.lower() in self.stop_words:
                continue

            # If the configuration is set to not include punctuation and the token is a punctuation mark, skip it
            if not self.config.include_punctuation and not any(char.isalnum() for char in token):
                continue

            # If the token is a number and its length is less than 5, skip it
            if token.isdigit() and len(token) < 5:
                continue

            # If the configuration is set to use stemming or lemmatization, apply the stemming or lemmatization to the token
            if self.config.use_stem and self.stemmer:
                token = self.stemmer.stem(token)
            elif self.config.use_lemmatize and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)

            # Add the processed token to the list of processed tokens
            processed_tokens.append(token)

        # Return the list of processed tokens
        return processed_tokens

    def detectCharacterNames(self, tokens: List[str]) -> List[Tuple[str, int]]:
        """
        Detect the names of characters in the given list of tokens.

        This method first counts the frequency of each word in the given list of tokens.
        It then loops through the words and their frequencies and checks if the word meets the following conditions:
            - If the frequency of the word is greater than 2.
            - If the length of the word is greater than 2.
            - If the first character of the word is uppercase.
            - If the word is not a number.
            - If the lower case version of the word is not in the list of character indicators.
            - If the lower case version of the word is not in the list of stop words.

        If the word meets all the conditions, it adds the word and its frequency to the list of name candidates.
        Finally, it returns the list of name candidates sorted by their frequency in descending order, with the top k names returned.

        Args:
            tokens (List[str]): The list of tokens to detect character names from.

        Returns:
            List[Tuple[str, int]]: The list of character names and their frequencies.
        """
        if not self.config.detect_characters:
            return []

        word_freq = Counter()
        name_candidates = list()

        for word, freq in word_freq.items():
            if (freq > 2 and
                len(word) > 2 and
                word[0].isupper() and
                not word.isdigit() and
                word.lower() not in self.character_indicators and
                word.lower() not in self.stop_words):
                name_candidates.append((word, freq))

        return sorted(name_candidates, key=lambda x:x[1], reverse=True)[:self.config.top_k]

    def _extractDialogueSections(self, content: str) -> List[str]:
        """
        Extract and return all the dialogue sections from the given content.

        This method first extracts all the dialogue sections from the content using several regular expression patterns.
        It then loops through all the extracted dialogue sections and sanitizes them by removing any leading or trailing whitespace.
        If the sanitized dialogue section is not empty and has a length of at least 3, it adds it to the list of all dialogue sections.
        Finally, it loops through the list of all dialogue sections and adds only the unique dialogue sections to the list of unique dialogue sections.

        Args:
            content (str): The content to extract the dialogue sections from.

        Returns:
            List[str]: The list of unique dialogue sections.
        """

        all_dialogues = list()
        unique_dialogues = list()
        seen = set()

        # Define the regular expression patterns to match the dialogue sections
        dialogue_patterns = list([
            # Pattern to match a dialogue section enclosed in double quotes
            r'"[^"\\]*(?:\\.[^"\\]*)*"',
            # Pattern to match a dialogue section enclosed in single quotes
            r"'[^'\\]*(?:\\.[^'\\]*)*'",

            # Pattern to match a dialogue section that starts with a character's name followed by a verb and then enclosed in double quotes
            r'(said|asked|replied|exclaimed|shouted|whispered|muttered|cried|answered|continued|added|observed|remarked|declared|announced|stated),\s*"[^"]*"',
            # Pattern to match a dialogue section that starts with a character's name followed by a verb and then enclosed in single quotes
            r'"[^"]*"\s*(said|asked|replied|exclaimed|shouted|whispered|muttered|cried|answered|continued|added|observed|remarked|declared|announced|stated)',

            # Pattern to match a dialogue section that starts with a character's name followed by a verb and then enclosed in double quotes
            r'(?:\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|asked|replied|exclaimed|shouted|whispered|muttered|cried|answered|continued|added|observed|remarked|declared|announced|stated)),\s*"[^"]*"',
        ])

        # Define the regular expression pattern to match a simple dialogue section enclosed in double quotes
        simple_dialogue_pattern = r'"[^"\\]*(?:\\.[^"\\]*)*"'

        # Extract all the dialogue sections from the content using the regular expression patterns
        for pattern in dialogue_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if isinstance(match, tuple):
                    dialogue_part = match[-1] if isinstance(match[-1], str) else match
                else:
                    dialogue_part = match

                # Sanitize the dialogue section by removing any leading or trailing whitespace
                sanitized_dialogue = self._sanitizeTxt(str(dialogue_part))
                if sanitized_dialogue and len(sanitized_dialogue) >= 3:
                    # Add the sanitized dialogue section to the list of all dialogue sections
                    all_dialogues.append(sanitized_dialogue)

        # If no dialogue sections were found using the regular expression patterns, try to extract them using a simple regular expression pattern
        if not all_dialogues:
            simple_quotes = re.findall(simple_dialogue_pattern, content)
            for quote in simple_quotes:
                # Sanitize the dialogue section by removing any leading or trailing whitespace
                sanitized_quote = self._sanitizeTxt(quote)
                if sanitized_quote and len(sanitized_quote) >= 3:
                    # Add the sanitized dialogue section to the list of all dialogue sections
                    all_dialogues.append(sanitized_quote)

        # Loop through the list of all dialogue sections and add only the unique dialogue sections to the list of unique dialogue sections
        for dialogue in all_dialogues:
            if dialogue not in seen:
                unique_dialogues.append(dialogue)
                seen.add(dialogue)

        return unique_dialogues

    def analyzeDialoguePatterns(self, content:str) -> Dict[str, Any]:
        """
        Analyze the given content for dialogue patterns.

        This method takes a string as input and returns a dictionary containing statistics about the dialogue patterns in the content.

        The returned dictionary contains the following keys:

        - total_dialogues: The total number of dialogue sections found in the content.
        - average_dialogue_length_char: The average length of a dialogue section in characters.
        - average_dialogue_length_word: The average length of a dialogue section in words.
        - total_dialogue_words: The total number of words found in all dialogue sections.
        - common_dialogue_words: A list of the top k most common words found in all dialogue sections.
        - dialogue_verb_usage: A dictionary containing the frequency of each verb found in the dialogue sections.
        - dialogue_examples: A list of up to 5 examples of dialogue sections found in the content.

        If the configuration does not specify to assess dialogue patterns, an empty dictionary is returned.
        """
        if not self.config.assess_dialogue_patterns:
            return dict()

        # Extract all the dialogue sections from the content
        dialogues = self._extractDialogueSections(content)

        # Initialize variables to store the total number of words and characters in all dialogue sections
        dialogue_words = list()
        dialogue_verbs = dict()
        dialogue_words_cnt = 0
        dialogue_chars_cnt = 0

        # Loop through all the dialogue sections and extract the words and characters
        for dialogue in dialogues:
            # Preprocess the dialogue section to extract the words
            dialogue_token = self.preprocessTxt(dialogue)
            dialogue_words.extend(dialogue_token)

            # Increment the total number of words and characters
            dialogue_words_cnt += len(dialogue.split())
            dialogue_chars_cnt += len(dialogue)

        # Calculate the frequency of each word in all dialogue sections
        dialogue_word_freq = Counter(dialogue_words)

        # Find the top k most common words in all dialogue sections
        common_dialogue_words = dialogue_word_freq.most_common(self.config.top_k)

        # Loop through all the dialogue sections and find the frequency of each verb
        for verb in self.dialogue_verbs:
            if verb in dialogue_word_freq:
                dialogue_verbs[verb] = dialogue_word_freq[verb]
            elif any(verb in dialogue.lower() for dialogue in dialogues):
                dialogue_verbs[verb] = sum(dialogue.lower().count(verb) for dialogue in dialogues)

        # Return the statistics about the dialogue patterns
        return {
            'total_dialogues': len(dialogues),
            'average_dialogue_length_char': dialogue_chars_cnt / len(dialogues) if dialogues else 0,
            'average_dialogue_length_word': dialogue_words_cnt / len(dialogues) if dialogues else 0,
            'total_dialogue_words': dialogue_words_cnt,
            'common_dialogue_words': common_dialogue_words,
            'dialogue_verb_usage': dialogue_verbs,
            'dialogue_examples': dialogues[:5] if dialogues else list()
        }

class CorpusAnalyzer():
    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.processor = TextProcessor(config)

    def _readFile(self, filePath: Path) -> str:
        """
        Reads a file at the specified path and returns its contents as a string.

        This function first attempts to read the file with the following encodings in order:
            - utf-8
            - latin-1
            - cp1252
            - iso-8859-1
            - utf-16

        If the file cannot be read with any of the above encodings, it will attempt to read the file with
        utf-8 encoding and ignore any errors that may occur. This may result in some characters being lost.

        If the file still cannot be read, a UnicodeDecodeError is raised with a message indicating the problem.
        """
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'utf-16']

        for encoding in encodings:
            try:
                # Attempt to read the file with the current encoding
                with open(file=filePath, mode='r', encoding=encoding) as f:
                    content = f.read()

                # If the reading was successful, log a message and return the contents
                logger.info(f'Reading file ({filePath.name}) successful with {encoding} encoding')
                return content

            except UnicodeEncodeError:
                # If the reading failed with the current encoding, continue to the next encoding
                continue

            except Exception as e:
                # If an exception occurred while reading the file, log a warning and continue to the next encoding
                logger.warning(f'Failed to read {filePath.name} with {encoding} encoding: {e}')
                continue

        try:
            # If all the above encodings failed, attempt to read the file with utf-8 encoding and ignore any errors
            with open(file=filePath, mode='r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # If the reading was successful, log a warning and return the contents
            logger.warning(f'Reading file ({filePath.name}) with error handling (some characters may be lost)')
            return content

        except Exception as e:
            # If the file still cannot be read, raise a UnicodeDecodeError with a message indicating the problem
            raise UnicodeDecodeError('Unknown', b'', 0, 0, f'Could not decode file ({filePath.name}) with any common encodings: {e}')

    def loadCorpus(self, directory:str) -> Dict[str, Dict[str, Any]]:
        """
        Loads a corpus (a collection of text files) from a given directory.

        This function first checks if the given directory exists. If not, it raises a FileNotFoundError.

        It then iterates through all the files in the directory and checks if they are supported text files (TXT or MD).
        If a file is supported, it attempts to read the file and preprocess its contents.

        If the preprocessing is successful, it extracts various statistics from the file such as the number of tokens, sentences, and detected character names.
        It also analyzes the dialogue patterns in the file and stores the results in a dictionary.

        Finally, it returns a dictionary containing the preprocessed corpus data.

        Args:
            directory (str): The path to the directory containing the corpus.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary containing the preprocessed corpus data.
        """
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f'Directory {directory} not found')

        supported_extensions = {'.txt', '.md'}
        corpus_data = dict()
        found_files = list()
        process_count = 0

        # Iterate through all the files in the directory
        for file_path in path.glob(pattern='*'):
            if file_path.is_file():
                found_files.append(file_path)
                logger.debug(f'Found files: {file_path.name} (size: {file_path.stat().st_size} bytes)')

        logger.info(f'Total found files: {len(found_files)}')

        # Iterate through all the found files and preprocess them
        for file_path in found_files:
            if file_path.suffix.lower() in supported_extensions:
                try:
                    # Read the file contents
                    content = self._readFile(filePath=file_path)
                    file_size = len(content)

                    logger.info(f'Preprocessing {file_path.name} ({file_size} bytes)')

                    # If the file is larger than the minimum file size, preprocess it
                    if content and file_size >= self.config.min_file_size:
                        # Tokenize the file contents
                        sentences = self.processor.tokenizer.tokenizeSentence(content)
                        dialogue_analysis = self.processor.analyzeDialoguePatterns(content)

                        # Preprocess the file contents
                        preprocessed_tokens = self.processor.preprocessTxt(content)
                        character_names = self.processor.detectCharacterNames(preprocessed_tokens)

                        # Store the preprocessed corpus data
                        corpus_data[file_path.name] = {
                            'tokens': preprocessed_tokens,
                            'dialogue_analysis': dialogue_analysis,
                            'character_names': character_names,
                            'file_stats': {
                                'file_size': file_size,
                                'token_count': len(preprocessed_tokens),
                                'sentences': sentences
                            }
                        }
                        process_count += 1
                        logger.info(f'Analyzed {file_path.name}: {len(preprocessed_tokens)} tokens, '
                                    f'{len(character_names)} candidates for characters, '
                                    f'{len(dialogue_analysis.get("total_dialogues", 0))} dialogues')
                    else:
                        logger.warning(f'Skipped {file_path.name}: too small ({file_size} bytes)')

                except Exception as e:
                    logger.error(f'Error preprocessing {file_path.name}: {e}')

            else:
                logger.debug(f'Skipping due to unsupported file: {file_path.name}')

        # If no corpus data was found, raise an error
        if not corpus_data:
            all_files = list(path.iterdir())
            file_list = '\n '.join([file.name for file in all_files if file.is_file()])
            subfolder_list = '\n '.join([folder.name for folder in all_files if folder.is_dir()])

            err_msg = f'No suitable narrative files found in {directory}'
            if file_list:
                err_msg += f'Found files:\n {file_list}'
            if found_files:
                err_msg += f'Found subfolders:\n {subfolder_list}'

            raise ValueError(err_msg)

        logger.info(f'Successfully processed {process_count} files')
        return corpus_data

class NGramsAnalyzer():
    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.corpus_analyzer = CorpusAnalyzer(config)

    def _composeNGrams(self, tokens: List[str], n:int) -> List[Tuple[str, ...]]:
        """
        Compose a list of n-grams from the given list of tokens.

        Args:
            tokens (List[str]): The list of tokens to compose n-grams from.
            n (int): The length of n-grams to compose.

        Returns:
            List[Tuple[str, ...]]: A list of n-grams composed from the given list of tokens.
        """
        return list() if len(tokens) < n else list(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

    def _calcDocFreq(self, documents: Dict[str, List[str]]) -> Dict[Tuple[str, ...], int]:
        """
        Calculate the document frequency of each n-gram in the corpus.

        Args:
            documents (Dict[str, List[str]]): A dictionary where the key is the file name
                and the value is a list of tokens in the file.

        Returns:
            Dict[Tuple[str, ...], int]: A dictionary where the key is an n-gram and the value
                is the number of documents that contain the n-gram.
        """
        doc_freq = Counter()

        # Iterate over each document in the corpus
        for fname, tokens in documents.items():
            # Calculate the n-grams for the current document
            doc_ngrams = set(self._composeNGrams(tokens, self.config.n))
            # Iterate over each unique n-gram in the document
            for ngram in doc_ngrams:
                # Increment the document frequency of the n-gram
                doc_freq[ngram] += 1

        return doc_freq

    def _aggerateDialogueVerbs(self, corpus_data: Dict[str, Any]) -> List[Tuple[str, int]]:
        """
        Aggregate the dialogue verb usage from each document in the corpus.

        Args:
            corpus_data (Dict[str, Any]): A dictionary where the key is the file name and
                the value is a dictionary containing the analysis data for the file.

        Returns:
            List[Tuple[str, int]]: A list of tuples where the first element is the
                dialogue verb and the second element is the count of occurrences in the corpus.
        """
        verb_counter = Counter()

        # Iterate over each document in the corpus
        for data in corpus_data.values():
            # Get the dialogue verb usage data for the current document
            dialogue_verbs = data['dialogue_analysis'].get('dialogue_verb_usage', {})
            # Iterate over each dialogue verb and its count in the document
            for verb, cnt in dialogue_verbs.items():
                # Increment the count of the dialogue verb in the corpus
                verb_counter[verb] += cnt

        # Return the top k dialogue verbs in the corpus
        return verb_counter.most_common(self.config.top_k)

    def _getMostCommonNgrams(self, ngrams_counter: Dict[Tuple[str, ...], int]):
        """
        Given a counter of n-grams and their frequencies, returns a list of the
        most common n-grams in the corpus along with their frequencies and percentages.

        Args:
            ngrams_counter (Dict[Tuple[str, ...], int]): A dictionary where the key is an n-gram
                and the value is the frequency of the n-gram in the corpus.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary contains the
                n-gram, its frequency, and its percentage in the corpus.
        """
        most_common = list()
        total_freq = sum(ngrams_counter.values())

        # Iterate over the n-grams sorted by their frequency in descending order
        for ngram, freq in Counter(ngrams_counter).most_common(self.config.top_k):
            # Join the n-gram tokens into a single string
            ngram_txt = ' '.join(ngram)
            # Append the n-gram data to the list
            most_common.append({
                'ngram': ngram_txt,
                'frequency': freq,
                'percentage': (freq / total_freq) * 100 if total_freq > 0 else 0
            })

        return most_common

    def _getDocFreqStats(self, doc_freq: Dict[Tuple[str, ...], int], total_docs:int) -> List[Dict[str, Any]]:
        """
        Given a dictionary of n-grams and their document frequencies, returns a list of dictionaries
        containing the n-gram, its document frequency, and its coverage percentage in the corpus.

        Args:
            doc_freq (Dict[Tuple[str, ...], int]): A dictionary where the key is an n-gram and the value is the number of documents that contain the n-gram.
            total_docs (int): The total number of documents in the corpus.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary contains the n-gram, its document frequency, and its coverage percentage in the corpus.
        """

        stats = list()
        for ngram, freq in Counter(doc_freq).most_common(self.config.top_k):
            # Join the n-gram tokens into a single string
            ngram_txt = ' '.join(ngram)
            # Append the n-gram data to the list
            stats.append({
                'ngram': ngram_txt,  # The n-gram
                'document_frequency': freq,  # The number of documents that contain the n-gram
                'coverage_percentage': (freq / total_docs) * 100 if total_docs > 0 else 0  # The coverage percentage of the n-gram in the corpus
            })

        return stats

    def _getCharNgrams(self, ngrams_counter: Dict[Tuple[str, ...], int], char_freq:Counter) -> List[Dict[str, Any]]:
        """
        Given a counter of n-grams and their frequencies, and a counter of character names and their frequencies,
        returns a list of dictionaries containing the character name, the n-gram, and the frequency of the n-gram.

        The list is sorted in descending order of the frequency of the n-grams, and only the top 15 n-grams are returned.

        Args:
            ngrams_counter (Dict[Tuple[str, ...], int]): A dictionary where the key is an n-gram and the value is the frequency of the n-gram in the corpus.
            char_freq (Counter): A counter of character names and their frequencies.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries where each dictionary contains the character name, the n-gram, and the frequency of the n-gram.
        """
        # Get the top k character names by frequency
        dominant_chars = [char for char, _ in char_freq.most_common(self.config.top_k)]

        # Initialize an empty list to store the character n-grams
        char_ngrams = list()

        # Iterate over each n-gram and its frequency
        for ngram, freq in ngrams_counter.items():
            # Join the n-gram tokens into a single string
            ngram_txt = ' '.join(ngram)

            # Iterate over each dominant character
            for char in dominant_chars:
                # Check if the character name is in the n-gram
                if char.lower() in ngram_txt:
                    # Append the character n-gram data to the list
                    char_ngrams.append({
                        'character': char,
                        'ngram': ngram_txt,
                        'frequency': freq
                    })
                    # Break out of the loop since we've found a match
                    break

        # Sort the character n-grams by their frequency in descending order
        return sorted(char_ngrams, key=lambda x: x['frequency'], reverse=True)[:15]

    def analyzeCorpus(self, directory: str) -> Dict[str, Any]:
        """
        Analyze a corpus of files and return a dictionary of results.

        The results will contain the following information:
        - Corpus overview: total files, total tokens, total characters, unique n-grams, filtered n-grams, analysis time
        - File statistics: for each file, the file size, token count, sentence count
        - Text analysis: most common characters, total dialogues found, common dialogue verbs
        - N-gram analysis: most common n-grams, document frequency, character specific n-grams

        :param directory: The directory to analyze
        :return: A dictionary of results
        """
        logger.info(f'Analyzing corpus: {directory}')

        start_time = time.time()

        try:
            # Initialize an empty list to store all tokens across the corpus
            all_tokens = list()

            # Initialize an empty list to store all character names across the corpus
            all_chars = list()

            # Initialize an empty dictionary to store the tokens for each document
            document_tokens = dict()

            # Initialize a counter to store the frequency of each character name
            char_freq = Counter()

            # Load the corpus data
            corpus_data = self.corpus_analyzer.loadCorpus(directory)

            # Iterate over each file in the corpus
            for filename, data in corpus_data.items():
                # Get the tokens for the current file
                tokens = data['tokens']

                # Add the tokens to the list of all tokens
                all_tokens.extend(tokens)

                # Store the tokens for the current file in the document tokens dictionary
                document_tokens[filename] = tokens

            # Log the total number of tokens across the corpus
            logger.info(f'Total tokens across corpus: {len(all_tokens)}')

            # Compose the n-grams from the tokens
            ngrams_list = self._composeNGrams(all_tokens, self.config.n)

            # Count the frequency of each n-gram
            ngrams_cnt = Counter(ngrams_list)

            # Filter out n-grams with frequency less than the minimum frequency
            filtered_ngrams = {
                ngram: freq for ngram, freq in ngrams_cnt.items() if freq >= self.config.min_freq
            }

            # Calculate the document frequency for each n-gram
            doc_freq = self._calcDocFreq(document_tokens)

            # Iterate over each file in the corpus
            for data in corpus_data.values():
                # Add the character names to the list of all character names
                all_chars.extend(data['character_names'])

            # Iterate over each character name and its frequency
            for name, freq in all_chars:
                # Increment the frequency of the character name in the character frequency counter
                char_freq[name] += freq

            # Calculate the total number of dialogues found across the corpus
            total_dialogue = sum(data['dialogue_analysis'].get('total_dialogue', 0) for data in corpus_data.values())

            # Create the results dictionary
            results = {
                'corpus_overview': {
                    'total_files': len(corpus_data),
                    'total_tokens': len(all_tokens),
                    'total_characters': len(all_chars),
                    'unique_ngrams': len(ngrams_cnt),
                    'filtered_ngrams': len(filtered_ngrams),
                    'analysis_time': time.time() - start_time,
                    'corpus_directory': directory
                },
                'file_statistics': {
                    filename: data['file_stats'] for filename, data in corpus_data.items()
                },
                'text_analysis': {
                    'most_common_characters': char_freq.most_common(self.config.top_k),
                    'total_dialogues_found':total_dialogue,
                    'common_dialogue_verbs': self._aggerateDialogueVerbs(corpus_data)
                },
                'ngram_analysis': {
                    'most_common_ngrams': self._getMostCommonNgrams(filtered_ngrams),
                    'document_frequency': self._getDocFreqStats(doc_freq, len(corpus_data)),
                    'character_specific_ngrams': self._getCharNgrams(filtered_ngrams, char_freq)
                }
            }

            # Return the results
            return results

        except Exception as e:
            logger.error(f'Analysis error: {e}')
            raise

class ResultsFormatter():
    @staticmethod
    def format(results: Dict[str, Any], formatType: str='text', filename: Optional[str]= None) -> str:
        """
        Formats the analysis results into a string based on the formatType.
        
        Parameters:
        results (Dict[str, Any]): The analysis results.
        formatType (str): The format type of the output. Can be 'json', 'csv', or 'text'.
        filename (Optional[str]): The file name to write the results to. If not provided, the results will not be written to a file.
        
        Returns:
        str: The formatted results string.
        """
        match formatType:
            case 'json':
                return ResultsFormatter._formatJson(results, filename)
            case 'csv':
                return ResultsFormatter._formatCsv(results, filename)
            case _:
                return ResultsFormatter._formatText(results, filename)

    @staticmethod
    def _formatJson(results: Dict[str, Any], filename: Optional[str]=None) -> str:
        """
        Formats the analysis results into a JSON string.

        Parameters:
        results (Dict[str, Any]): The analysis results.
        filename (Optional[str]): The file name to write the results to. If not provided, the results will not be written to a file.

        Returns:
        str: The formatted JSON results string.
        """
        json_output = json.dumps(results, indent=2, ensure_ascii=False)

        if filename:
            with open(file=filename, mode='w', encoding='utf-8') as f:
                f.write(json_output)
            logger.info(f'JSON results save to {filename}')

        return json_output

    @staticmethod
    def _formatCsv(results: Dict[str, Any], filename: Optional[str]=None) -> str:
        """
        Formats the analysis results into a CSV string.

        Parameters:
        results (Dict[str, Any]): The analysis results.
        filename (Optional[str]): The file name to write the results to. If not provided, the results will not be written to a file.

        Returns:
        str: The formatted CSV results string.
        """
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        writer.writerow(['type', 'rank', 'content', 'frequency', 'percentage'])

        ngrams = results['ngram_analysis']['most_common_ngrams']
        for i, ngram_data in enumerate(ngrams, 1):
            writer.writerow(['ngram', i, ngram_data['ngram'], ngram_data['frequency'], f"{ngram_data['percentage']:.2f}"])

        characters = results['text_analysis']['most_common_characters']
        for i, (char, freq) in enumerate(characters, 1):
            writer.writerow(['character', i, char, freq, ''])

        csv_output = output.getvalue()

        if filename:
            with open(file=filename, mode='w', encoding='utf-8', newline='') as f:
                f.write(csv_output)
            logger.info(f'CSV results saved to {filename}')

        return csv_output

    @staticmethod
    def _formatText(results: Dict[str, Any], filename: Optional[str]=None) -> str:
        """
        Formats the analysis results into a human-readable text string.

        Parameters:
        results (Dict[str, Any]): The analysis results.
        filename (Optional[str]): The file name to write the results to. If not provided, the results will not be written to a file.

        Returns:
        str: The formatted text results string.
        """

        # Initialize the output list
        output = list()

        # Add the header
        output.append('=' * 50 + '\nText Corpus Analysis Report\n' + '=' * 50)

        # Add the corpus overview section
        overview = results['corpus_overview']
        output.append('\nCorpus Overview:')
        output.append(f"  Directory: {overview.get('corpus_directory')}\n"
                      f"  Files analyzed: {overview.get('total_files')}\n"
                      f"  Total tokens: {overview.get('total_tokens')}\n"
                      f"  Unique n-grams: {overview.get('unique_ngrams')}\n"
                      f"  Analysis time: {overview.get('analysis_time')}")

        # Add the file statistics section
        file_stats = results['file_statistics']
        output.append('\n-- File Statistics --\n' + '-' * 40)
        for filename, stats in file_stats.items():
            output.append(f'  {filename:}\n'
                          f'  Size: {stats["file_size"]:,} characters, '
                          f'Tokens: {stats["token_count"]:,}, '
                          f'Sentences: {stats["sentences"]}')

        # Add the most frequent character names section
        characters = results['text_analysis']['most_common_characters']
        if characters:
            output.append(f'\n-- Most Frequent Character Names: --\n' + '-' * 40)
            for i, (char, freq) in enumerate(characters, 1):
                output.append(f'{i:2d}. {char:20} {freq:4d} mentions')
        else:
            output.append('\nNo character names detected (try --detect-characters flag)')

        # Add the most used dialogue verbs section
        dialogue_verbs = results['text_analysis']['common_dialogue_verbs']
        if dialogue_verbs:
            output.append(f'\n-- Most Used Dialogue Verbs: --\n' + '-' * 40)
            for i, (verb, count) in enumerate(dialogue_verbs, 1):
                output.append(f'{i:2d}. {verb:15} {count:4d}')

        # Add the most common N-Grams section
        ngrams = results['ngram_analysis']['most_common_ngrams']
        output.append(f'\n-- Top {len(ngrams)} most common N-Grams: --\n' + '-' * 40)
        for i, ngram_data in enumerate(ngrams, 1):
            output.append(f'{i:2d}. {ngram_data["ngram"]:35} {ngram_data["frequency"]:4d} ({ngram_data["percentage"]:5.2f}%)')

        # Add the character specific N-Grams section
        char_ngrams = results['ngram_analysis']['character_specific_ngrams']
        if char_ngrams:
            output.append('\n-- Character Specific N-Grams: --\n' + '-' * 40)
            current_char = ''
            for i, ngram_data in enumerate(char_ngrams, 1):
                if ngram_data['character'] != current_char:
                    current_char = ngram_data['character']
                    output.append(f'\n{current_char}:')
                output.append(f'  - {ngram_data["ngram"]:30} {ngram_data["frequency"]:3d}')

        # Join the output list into a single string
        text_output = '\n'.join(output)

        # If a filename is provided, write the output to the file
        if filename:
            with open(file=filename, mode='w', encoding='utf-8') as f:
                f.write(text_output)

            logger.info(f'Report saved to {filename}')

        return text_output

def initParser() -> argparse.ArgumentParser:
    """
    Initializes the command line parser.

    This function creates an ArgumentParser object with the following arguments:
        - corpus_dir: The directory containing the text corpus.
        - --n: The N-Gram size to analyze (default: 2).
        - --top-k: The number of most common N-Grams to report (default: 20).
        - --min-freq: The minimum frequency of N-Grams to report (default: 2).
        - --min-file-size: The minimum file size (in characters) to analyze (default: 100).
        - --no-stopwords: Disable the use of stopwords in the analysis.
        - --stem: Enable stemming in the analysis.
        - --lemmatize: Enable lemmatization in the analysis.
        - --case-sensative: Enable case sensitivity in the analysis.
        - --include-punctuation: Include punctuation in the analysis.
        - --detect-characters: Enable character detection in the analysis.
        - --assess-dialogue-patterns: Enable the assessment of dialogue patterns in the analysis.
        - --save-results: Save the analysis results to a file.
        - --verbose: Enable verbose logging.
        - --output-format: The output format of the analysis results (default: 'text', choices: ['text', 'json', 'csv']).

    Returns:
        An ArgumentParser object with the specified arguments.
    """
    parser = argparse.ArgumentParser(
        description='N-Gram Text Analyzer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
 # Basic analysis (works without NLTK punkt_tab)
 python app.py ./holmes

 # With character detection
 python app.py ./stories --detect-characters --min-freq 1

 # Verbose mode for debugging
 python app.py ./corpus --verbose""")
    parser.add_argument('corpus_dir', help='The directory containing the text corpus.')
    parser.add_argument('--n', type=int, default=2, help='The N-Gram size to analyze.')
    parser.add_argument('--top-k', type=int, default=20, help='The number of most common N-Grams to report.')
    parser.add_argument('--min-freq', type=int, default=2, help='The minimum frequency of N-Grams to report.')
    parser.add_argument('--min-file-size', type=int, default=100, help='The minimum file size (in characters) to analyze.')

    parser.add_argument('--no-stopwords', action='store_true', help='Disable the use of stopwords in the analysis.')
    parser.add_argument('--stem', action='store_true', help='Enable stemming in the analysis.')
    parser.add_argument('--lemmatize', action='store_true', help='Enable lemmatization in the analysis.')
    parser.add_argument('--case-sensative', action='store_true', help='Enable case sensitivity in the analysis.')
    parser.add_argument('--include-punctuation', action='store_true', help='Include punctuation in the analysis.')
    parser.add_argument('--detect-characters', action='store_true', help='Enable character detection in the analysis.')
    parser.add_argument('--assess-dialogue-patterns', action='store_true', help='Enable the assessment of dialogue patterns in the analysis.')
    parser.add_argument('--save-results', action='store_true', help='Save the analysis results to a file.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging.')
    parser.add_argument('--output-format', choices=['text', 'json', 'csv'], default='text', help='The output format of the analysis results.')

    return parser

def main():
    """
    Main function of the application.

    This function parses the command line arguments and configures the application
    according to the provided options. It then creates an instance of the NGramsAnalyzer
    class and calls its analyzeCorpus method to perform the analysis. Finally, it formats
    the results according to the specified output format and prints or saves them to a file.
    """
    obtainNLTK()
    parser = initParser()
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)

    try:
        # Create an instance of the AnalyzerConfig class based on the command line arguments
        config = AnalyzerConfig(
            n=args.n,
            top_k=args.top_k,
            min_freq=args.min_freq,
            use_stopwords=not args.no_stopwords,
            use_stem=args.stem,
            use_lemmatize=args.lemmatize,
            case_sensative=args.case_sensitive,
            include_punctuation=args.include_punctuation,
            detect_characters=args.detect_characters,
            min_file_size=args.min_file_size,
            output_format=args.output_format,
            save_results=args.save_results,
            assess_dialogue_patterns=args.assess_dialogue_patterns
        )

        # Create an instance of the NGramsAnalyzer class and call its analyzeCorpus method
        analyzer = NGramsAnalyzer(config)
        results = analyzer.analyzeCorpus(args.corpus_dir)

        # Format the results according to the specified output format
        filename = None
        if args.save_results:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f'analysis_n{args.n}_{timestamp}.{args.output_format}'

        output = ResultsFormatter().format(results, args.output_format, filename)

        # Print or save the results to a file
        if args.output_format == 'text' and not args.save_results:
            print(output)

        logger.info('Analysis completed')

    except Exception as e:
        # Log any application errors
        logger.error(f'Application error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()