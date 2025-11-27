import sys
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import time
import json
from pathlib import Path
from re import sub

import argparse
from markovify import Text

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('markov_app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class GeneralStats():
    gen_success: int=0
    gen_fail: int=0
    gen_total: int=0
    gen_chars: int=0
    gen_words: int=0
    gen_time: float=0.0

    def startTime(self):
        self.gen_time = time.time()

    def getElapsedTime(self):
        return time.time() - self.gen_time

class MarkovGen():
    def __init__(self, state_size: int=2) -> None:
        """
        Initializes the Markov Generation class.

        Args:
            state_size (int, optional): The number of words to look back when generating text. Defaults to 2.
        """
        self.state_size = state_size
        self.model: Optional[Text]
        self.stats = GeneralStats()

    def loadModel(self, filePath: str) -> None:
        """
        Loads a pre-trained Markov model from a JSON file.

        This function loads a pre-trained Markov model from a JSON file.
        The JSON file should contain the model's internal state, which is a
        dictionary mapping words to their next-word probabilities.

        Args:
            filePath (str): The path to the JSON file containing the model.

        Raises:
            FileNotFoundError: If the file at the given path does not exist.
            Exception: If an error occurs while loading the model.
        """
        try:
            # Open the file and load its contents into a dictionary
            with open(file=filePath, mode='r', encoding='utf-8') as f:
                model_json = json.load(f)

            # Create a new Text object from the loaded JSON
            self.model = Text.from_json(model_json)

            # Log a message to indicate that the model was loaded successfully
            logger.info(f'Model loaded successfully')

        except FileNotFoundError:
            # Log an error if the file does not exist
            logger.error(f'Error loading model: File not found at {filePath}')

        except Exception as e:
            # Log an error if an exception occurs while loading the model
            logger.error(f'Error loading model: {e}')

    def _preprocess(self, text: str) -> str:
        """
        Preprocesses a given text by replacing multiple spaces with a single space,
        replacing multiple newline characters with a single newline character, and
        adding a period at the end of each line if it does not already end with a
        sentence-ending punctuation mark.

        Args:
            text (str): The text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        # Replace multiple spaces with a single space
        text = sub(pattern=r'\s+', repl=' ', string=text)

        # Replace multiple newline characters with a single newline character
        text = sub(pattern=r'\n\s*\n', repl='\n\n', string=text)

        # Split the text into lines
        lines = text.splitlines()

        # Process each line
        processed_lines = list()
        for line in lines:
            # Strip leading and trailing whitespace from the line
            line = line.strip()

            # If the line does not end with a sentence-ending punctuation mark, add one
            if not line[-1] in '.?!':
                line += '.'
            processed_lines.append(line)

        # Join the processed lines back into a single string
        return '\n'.join(processed_lines)

    def loadTxt(self, filePath: str) -> str:
        """
        Attempts to load a text from a file at the specified path.

        This function first checks if the file exists. If not, it raises a FileNotFoundError.
        It then attempts to read the file with different encodings (utf-8, latin-1, and cp1252) until it succeeds.
        If it fails with all encodings, it raises a ValueError.

        Once the text is loaded, it preprocesses the text by replacing multiple spaces with a single space,
        replacing multiple newline characters with a single newline character, and adding a period at the end of each line
        if it does not already end with a sentence-ending punctuation mark.

        Args:
            filePath (str): The path to the file containing the text.

        Returns:
            str: The preprocessed text.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file could not be decoded with any common encoding.
        """
        try:
            # Check if the file exists
            path = Path(filePath)
            if not path.exists():
                raise FileNotFoundError(f'\nText file at the "{path}" not found')

            # Attempt to read the file with different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            text = None
            for encoding in encodings:
                try:
                    # Attempt to read the file with the current encoding
                    with open(file=path, mode='r', encoding=encoding) as f:
                        text = f.read()

                    # If the reading was successful, break the loop
                    break
                except UnicodeDecodeError:
                    # If the reading failed, continue to the next encoding
                    continue

            # If the text is still None after attempting to read with all encodings, raise a ValueError
            if text is None:
                raise ValueError('Unknown', b'', 0, 0, 'Could not decode file with any common encoding')

            # Preprocess the text
            text = self._preprocess(text)

            # Log the number of words and characters loaded
            logger.info(f'Loaded {len(text.split())} words, {len(text)} characters')

            # Return the preprocessed text
            return text

        except Exception as e:
            # Log an error if an exception occurs while loading the text
            logger.error(f'Error while loading text from file: {e}')
            # Raise the exception
            raise

    def trainModel(self, text: str, **kwargs) -> None:
        """
        Trains a Text model from the given text.

        Args:
            text (str): The text to train the model with.
            **kwargs: Additional keyword arguments to pass to the Text model.

        Raises:
            Exception: If an error occurs while training the model.
        """
        try:
            # Create a dictionary of training parameters
            train_params = {
                # Set the state size of the model to the instance variable
                'state_size': self.state_size,
                # Pass any additional keyword arguments to the Text model
                **kwargs
            }

            # Create a Text model with the training parameters
            self.model = Text(text, **train_params)

            # Log a success message if the model was trained successfully
            logger.info('Model trained successfully')

        except Exception as e:
            # Log an error message if an exception occurs while training the model
            logger.error(f'Error while training model: {e}')
            # Raise the exception
            raise

    def saveModel(self, savePath: str) -> None:
        """
        Saves the Text model to a file at the specified path.

        This function first checks if the model exists. If not, it raises a ValueError.
        It then attempts to convert the model to JSON format and save it to the specified file.
        If the saving is successful, it logs an information message with the path of the saved file.
        If an exception occurs while saving the model, it logs an error message with the exception and raises the exception.

        Args:
            savePath (str): The path to save the model to.

        Raises:
            ValueError: If the model does not exist.
            Exception: If an error occurs while saving the model.
        """
        if not self.model:
            raise ValueError('Model not found. Please, create a model first.')

        try:
            # Convert the model to JSON format
            model_json = self.model.to_json()

            # Save the model to the specified file
            with open(file=savePath, mode='w', encoding='utf-8') as f:
                # Use the json.dump function to save the model to the file
                # The indent parameter is set to 2 to format the JSON with 2 spaces of indentation
                json.dump(model_json, f, indent=2)

            # Log an information message with the path of the saved file
            logger.info(f'Model saved to: {savePath}')

        except Exception as e:
            # Log an error message with the exception if an error occurs while saving the model
            logger.error(f'Error while saving a model: {e}')
            # Raise the exception
            raise

    def _generateBounded(self, min_chars: Optional[int]=None, max_chars: Optional[int]=None, max_attempts: int=50, **kwargs) -> Optional[str]:
        """
        Attempts to generate a sentence with the given constraints for maximum and/or minimum characters.

        If the model is not found, it raises a ValueError.

        It takes in the following parameters:
            min_chars: The minimum number of characters in the sentence.
            max_chars: The maximum number of characters in the sentence.
            max_attempts: The maximum number of attempts to generate a sentence.
            **kwargs: Additional keyword arguments to be passed to the Text model.

        It first checks if the model exists. If not, it raises a ValueError.

        It then attempts to generate a sentence with the given parameters.
        If a sentence is generated, it checks if the sentence meets the given min and max character limits.
        If the sentence meets the limits, it logs a debug message with the number of characters in the sentence and returns the sentence.

        If the sentence does not meet the limits, it logs a debug message with the reason for rejection and continues to the next attempt.

        If an exception occurs while generating a sentence, it logs a warning message with the exception and continues to the next attempt.

        If all attempts fail, it logs a warning message with the number of attempts and returns None.
        """

        # Check if the model exists
        if not self.model:
            raise ValueError('Model not found. Please, create a model first.')

        # Set default parameters
        default_params = {
            # The number of attempts to generate a sentence
            'tries': 50,
            # The maximum overlap ratio between adjacent words
            'max_overlap_ratio': 0.7,
            # The maximum number of overlapping characters between adjacent words
            'max_overlap_total': 15,
            # The maximum number of characters in the sentence
            'max_chars': None,
            # The minimum number of characters in the sentence
            'min_chars': None,
            # Whether to test the output of the generated sentence
            'test_output': True
        }
        default_params.update(kwargs)

        # If min_chars is given, estimate the minimum number of words
        if min_chars and not default_params['min_words']:
            estimated_min_words = max(1, min_chars // 6)
            default_params['min_words'] = estimated_min_words

        # If max_chars is given, estimate the maximum number of words
        if max_chars and not default_params['max_words']:
            estimated_max_words = max(2, max_chars // 4)
            default_params['max_words'] = estimated_max_words

        # Attempt to generate a sentence
        for attempt in range(max_attempts):
            try:
                sentence = self.model.make_sentence(default_params)
                if sentence:
                    # Count the number of characters in the sentence
                    char_count = len(sentence)
                    # Check if the sentence meets the given min and max character limits
                    meets_min = min_chars is None or min_chars <= char_count
                    meets_max = max_chars is None or max_chars >= char_count

                    # If the sentence meets the limits, log a debug message and return the sentence
                    if meets_min and meets_max:
                        logger.debug(f'Generated sentence with {char_count} characters (attempt {attempt + 1})')
                        self.stats.gen_success += 1
                        self.stats.gen_words += len(sentence.split())
                        self.stats.gen_chars += len(sentence)

                        return sentence
                    else:
                        # If the sentence does not meet the limits, log a debug message with the reason for rejection
                        logger.debug(f'Sentence reject: {char_count} characters (min={min_chars}, max={max_chars})')

                # Increment the generation failure count
                self.stats.gen_fail += 1
            except Exception as e:
                # If an exception occurs while generating a sentence, log a warning message with the exception
                logger.warning(f'Generation attempt {attempt +1} failed: {e}')
                self.stats.gen_fail += 1

        # If all attempts fail, log a warning message with the number of attempts and return None
        logger.warning(f'Failed to generate suitable sentence after {max_attempts} attempts for max chars = {max_chars}, min chars = {min_chars}')
        return None

    def composeSentence(self, max_attempts: int=10, **kwargs) -> Optional[str]:
        """
        Compose a sentence using the Text model.

        This function takes in the following parameters:
            max_attempts: The maximum number of attempts to generate a sentence.
            **kwargs: Additional keyword arguments to be passed to the Text model.

        It first checks if the model exists. If not, it raises a ValueError.

        It then attempts to generate a sentence with the given parameters.
        If a sentence is generated, it logs a debug message with the number of characters in the sentence and returns the sentence.

        If the sentence does not meet the given min and max character limits, it logs a debug message with the reason for rejection and continues to the next attempt.

        If an exception occurs while generating a sentence, it logs a warning message with the exception and continues to the next attempt.

        If all attempts fail, it logs a warning message with the number of attempts and returns None.
        """

        if not self.model:
            raise ValueError('Model not found. Please, create a model first.')

        min_chars = kwargs.pop('min_chars', None)
        max_chars = kwargs.pop('max_chars', None)

        if min_chars is not None or max_chars is not None:
            # If min_chars and/or max_chars is given, generate a sentence bounded by the given limits
            return self._generateBounded(
                min_chars=min_chars,
                max_chars=max_chars,
                max_attempts=max_attempts,
                **kwargs
            )

        default_params = {
            'tries': 50, # The number of attempts to generate a sentence
            'max_overlap_ratio': 0.7, # The maximum overlap ratio between adjacent words
            'max_overlap_total': 15, # The maximum number of overlapping characters between adjacent words
            'max_words': None, # The maximum number of words in the sentence
            'min_words': None, # The minimum number of words in the sentence
            'test_output': True # Whether to test the output of the generated sentence
        }
        default_params.update(kwargs)

        for attempt in range(max_attempts):
            try:
                # Attempt to generate a sentence
                sentence = self.model.make_sentence(**default_params)
                if sentence:
                    # If a sentence is generated, log a debug message with the number of characters in the sentence
                    logger.debug(f'Generated sentence with {len(sentence)} characters (attempt {attempt + 1})')
                    self.stats.gen_success += 1
                    self.stats.gen_words += len(sentence.split())
                    self.stats.gen_chars += len(sentence)

                    # Return the generated sentence
                    return sentence

                # Increment the generation failure count
                self.stats.gen_fail += 1
            except Exception as e:
                # If an exception occurs while generating a sentence, log a warning message with the exception
                logger.warning(f'Generation attempt {attempt + 1} failed: {e}')
                self.stats.gen_fail += 1

        # If all attempts fail, log a warning message with the number of attempts and return None
        logger.warning(f'Failed to generate sentence after {max_attempts} attempts')
        return None

    def _composeShort(self, maxWords: int=10, maxChars: Optional[int]=None, maxTry: int=20) -> Optional[str]:
        return self.composeSentence(
            max_attempts=maxTry,
            max_words=maxWords,
            max_chars=maxChars,
            tries=100
        )

    def composeShorts(self, count: int=5, maxWords: int=10, maxChars: Optional[int]=None) -> List[str]:
        """
        Compose a specified number of short sentences with the given constraints for maximum and/or minimum characters.

        Parameters
        ----------
        count : int, optional
            The number of short sentences to generate (default is 5).
        maxWords : int, optional
            The maximum number of words in the sentence (default is 10).
        maxChars : Optional[int], optional
            The maximum number of characters in the sentence (default is None).

        Returns
        -------
        List[str]
            A list of generated short sentences.
        """
        sentences = list()

        # Start the timer
        self.stats.startTime()

        # Generate the specified number of short sentences
        for i in range(count):
            # Attempt to generate a short sentence
            sentence = self._composeShort(maxWords=maxWords, maxChars=maxChars)

            if sentence:
                # If a sentence is generated, log a debug message with the number of characters in the sentence
                logger.debug(f'Generated short sentence with {len(sentence)} characters (attempt {i+1})')
                sentences.append(sentence)
                self.stats.gen_total += 1
                words_count = len(sentence.split())
                chars_count = len(sentence)
                logger.info(f'Generated short sentence {i+1}/{count}: {words_count} words, {chars_count} characters')
            else:
                # If the sentence does not meet the given min and max character limits, log a debug message with the reason for rejection
                logger.error(f'Failed to generate short sentence {i+1}')

        # Get the elapsed time
        elapsed_time = self.stats.getElapsedTime()
        logger.info(f'Generated {len(sentences)} sentences in {elapsed_time:.2f} seconds')

        return sentences

    def _composeLong(self, minWords: int=15, minChars: Optional[int]=None, maxTry:int=20) -> Optional[str]:
        return self.composeSentence(
            max_attempts=maxTry,
            min_words=minWords,
            min_chars=minChars,
            tries=100
        )

    def composeLongs(self, count: int=5, minWords: int=15, minChars: Optional[int]=None) -> List[str]:
        """
        Compose a specified number of long sentences with the given constraints for minimum and/or maximum characters.

        This function generates a specified number of sentences with a minimum number of words and/or characters.
        It first starts a timer, then generates the specified number of long sentences.
        If a sentence is generated, it logs a debug message with the number of characters in the sentence.
        If the sentence does not meet the given min and max character limits, it logs a debug message with the reason for rejection.
        Finally, it logs the elapsed time and returns the list of generated long sentences.
        """
        sentences = list()
        self.stats.startTime()  # Start the timer

        # Generate the specified number of long sentences
        for i in range(count):
            # Attempt to generate a long sentence
            sentence = self._composeLong(minWords=minWords, minChars=minChars)

            if sentence:
                # If a sentence is generated, log a debug message with the number of characters in the sentence
                logger.debug(f'Generated long sentence with {len(sentence)} characters (attempt {i+1})')
                sentences.append(sentence)
                self.stats.gen_total += 1
                words_count = len(sentence.split())
                chars_count = len(sentence)
                logger.info(f'Generated long sentence {i+1}/{count}: {words_count} words, {chars_count} characters')
            else:
                # If the sentence does not meet the given min and max character limits, log a debug message with the reason for rejection
                logger.warning(f'Failed to generate long sentence {i + 1}')

        # Get the elapsed time
        elapsed_time = self.stats.getElapsedTime()
        logger.info(f'Generated {len(sentences)} sentences in {elapsed_time:.2f} seconds')

        return sentences

    def composeMulti(self, count: int=5, **kwargs) -> List[str]:
        """
        Compose a specified number of sentences with the given constraints.

        This function generates a specified number of sentences with the given constraints.
        It first starts a timer, then generates the specified number of sentences.
        If a sentence is generated, it logs a debug message with the number of characters in the sentence.
        If the sentence does not meet the given min and max character limits, it logs a debug message with the reason for rejection.
        Finally, it logs the elapsed time and returns the list of generated sentences.
        """
        sentences = []  # List to store the generated sentences

        self.stats.startTime()  # Start the timer

        # Generate the specified number of sentences
        for i in range(count):
            # Attempt to generate a sentence
            sentence = self.composeSentence(**kwargs)

            if sentence:
                # If a sentence is generated, log a debug message with the number of characters in the sentence
                sentences.append(sentence)
                self.stats.gen_total += 1
                words_count = len(sentence.split())
                chars_count = len(sentence)
                logger.info(f'Generated sentence {i+1}/{count}: {words_count} words, {chars_count} characters')
            else:
                # If the sentence does not meet the given min and max character limits, log a debug message with the reason for rejection
                logger.warning(f'Failed to generate sentence {i + 1}')

        # Get the elapsed time
        elapsed_time = self.stats.getElapsedTime()
        logger.info(f'Generated {len(sentences)} sentences in {elapsed_time:.2f} seconds')

        return sentences

    def printStatistics(self) -> None:
        """
        Prints out the generation statistics.

        The generation statistics include:
            - Total sentences generated
            - Successful generation
            - Failed generation
            - Total words
            - Total characters
            - Average sentence length (if successful generation > 0)
            - Success rate (if successful generation + failed generation > 0)
        """
        print('\n' + '=' * 50 + '\nGeneration Statistics\n' + '=' * 50)
        print(f'Total sentences generated: {self.stats.gen_total}\n\
Successful generation: {self.stats.gen_success}\n\
Failed generation: {self.stats.gen_fail}\n\
Total words: {self.stats.gen_words} words\n\
Total number of words in all generated sentences\n\
Total characters: {self.stats.gen_chars} characters\n' +'-' * 50)

        if self.stats.gen_success > 0:
            avg_words = self.stats.gen_words / self.stats.gen_success
            avg_chars = self.stats.gen_chars / self.stats.gen_success
            print(f'Average sentence length: {avg_words:.1f} words, {avg_chars:.1f} characters')

        if (self.stats.gen_success + self.stats.gen_fail) > 0:
            success_rate = self.stats.gen_success / (self.stats.gen_success + self.stats.gen_fail) * 100
            print(f'Success rate: {success_rate:.2f}%')

    def getModelInfo(self) -> Dict[str, Any]:
        if not self.model:
            return dict()

        return dict({
            'state_size': self.state_size,
            'parsed_sentences': len(self.model.parsed_sentences) if hasattr(self.model, 'parsed_sentences') else 0,
            'chain_size': len(self.model.chain.model) if hasattr(self.model.chain, 'model') else 0
        })

def initParser() -> argparse.ArgumentParser:
    """
    Initializes the command line parser.

    This function creates an ArgumentParser object with the following arguments:
        - file: Input text file
        - --count (-c): Number of sentences to generate (default: 5)
        - --state-size (-s): Markov state size (default: 2)
        - --max-words: Maximum words per sentence
        - --min-words: Minimum words per sentence
        - --max-chars: Maximum characters per sentence
        - --min-chars: Minimum characters per sentence
        - --save-model: Save trained model to a file
        - --load-model: Load a pre-trained model from json file
        - --short: Generate short sentences (max: 10 words)
        - --long: Generate long sentences (min: 15 words)
        - --verbose: Enable verbose logging

    Returns:
        An ArgumentParser object with the specified arguments.
    """
    parser = argparse.ArgumentParser(
        description='Markov Chain Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
=== Examples ===
# Basic usage
python app.py sample.txt

# Generate 10 sentences with state size 3
python app.py sample.txt --count 10 --state-size 3

# Character constraints only
python app.py sample.txt --min-chars 50 --max-chars 200

# Mixed word and character constraints
python app.py sample.txt --min-words 5 --max-words 12 --max-chars 120

# Short sentences with character limit
python app.py sample.txt --short --max-chars 100

# Long sentences with minimum character count
python app.py sample.txt --long --min-chars 150

# Complex constraints
python app.py sample.txt --min-words 8 --max-words 15 --min-chars 80 --max-chars 180

# Save and load models
python app.py sample.txt --save-model my_model.json
python app.py sample.txt --load-model my_model.json --count 5""")

    # Input text file
    parser.add_argument('file', help='Input text file')

    # Number of sentences to generate
    parser.add_argument('--count', '-c', type=int, default=5, help='Number of sentences to generate (default: 5)')

    # Markov state size
    parser.add_argument('--state-size', '-s', type=int, default=2, help='Markov state size (default: 2)')

    # Word constraints
    parser.add_argument('--max-words', type=int, help='Maximum words per sentence')
    parser.add_argument('--min-words', type=int, help='Minimum words per sentence')

    # Character constraints
    parser.add_argument('--max-chars', type=int, help='Maximum characters per sentence')
    parser.add_argument('--min-chars', type=int, help='Minimum characters per sentence')

    # Model saving and loading
    parser.add_argument('--save-model', type=str, help='Save trained model to a file')
    parser.add_argument('--load-model', type=str, help='Load a pre-trained model from json file')

    # Sentence length constraints
    parser.add_argument('--short', action='store_true', help='Generate short sentences (max: 10 words)')
    parser.add_argument('--long', action='store_true', help='Generate long sentences (min: 15 words)')

    # Verbose logging
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    return parser


def main():
    # Parse command line arguments
    parser = initParser()
    args = parser.parse_args()

    # If verbose logging is enabled, set the logging level to DEBUG
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize the Markov generator
        generator = MarkovGen(state_size=args.state_size)

        # If a model file is specified, load the pre-trained model from the file
        if args.load_model:
            generator.loadModel(args.load_model)
            print(f'Model loaded from {args.load_model}')
        else:
            # Load the input text file
            text = generator.loadTxt(args.file)

            # Train the Markov model
            generator.trainModel(text)

            # If a model file is specified, save the trained model to the file
            if args.save_model:
                generator.saveModel(args.save_model)
                print(f'Model saved to {args.save_model}')

        print('\n' + '=' * 50 + '\nGenerated Text\n' + '=' * 50)

        # Initialize the list to store the generated sentences
        sentences = list()

        # If generating short sentences, compose sentences with the given constraints
        if args.short:
            max_words = args.max_words or 10
            max_chars = args.max_chars
            constraint_desc = f'max {max_words} words'
            if max_chars:
                constraint_desc += f', max {max_chars} characters'

            print(f'- Generating {args.count} short sentences ({constraint_desc})')
            sentences = generator.composeShorts(
                count=args.count,
                maxWords=max_words,
                maxChars=max_chars
            )

        # If generating long sentences, compose sentences with the given constraints
        elif args.long:
            min_words = args.min_words or 15
            min_chars = args.min_chars
            constraint_desc = f'min {min_words} words'
            if min_chars:
                constraint_desc += f', min {min_chars} characters'
            print(f'- Generating {args.count} long sentences ({constraint_desc})')
            sentences = generator.composeLongs(
                count=args.count,
                minWords=min_words,
                minChars=min_chars
            )

        # If generating sentences with custom constraints, compose sentences with the given constraints
        else:
            gen_kwargs = dict()
            constraint_desc_list = list()

            if args.max_words: # If a maximum number of words is specified, add it to the constraints
                gen_kwargs['max_words'] = args.max_words
                constraint_desc_list.append(f'max {args.max_words} words')
            if args.min_words: # If a minimum number of words is specified, add it to the constraints
                gen_kwargs['min_words'] = args.min_words
                constraint_desc_list.append(f'min {args.min_words} words')
            if args.max_chars: # If a maximum number of characters is specified, add it to the constraints
                gen_kwargs['max_chars'] = args.max_chars
                constraint_desc_list.append(f'max {args.max_chars} characters')
            if args.min_chars: # If a minimum number of characters is specified, add it to the constraints
                gen_kwargs['min_chars'] = args.min_chars
                constraint_desc_list.append(f'min {args.min_chars} characters')

            if constraint_desc_list: # If there are any constraints, compose sentences with the given constraints
                constraint_desc = ', '.join(constraint_desc_list)
                print(f'- Generating {args.count} sentences ({constraint_desc})')
            else:
                print(f'- Generating {args.count} sentences...')

            sentences = generator.composeMulti(
                count=args.count,
                **gen_kwargs
            )

        if sentences: # If sentences were generated, print them out
            for i, sentence in enumerate(sentences, 1):
                word_count = len(sentence.split())
                char_count = len(sentence)

                print(f'\n{i}. {sentence}\n   (words: {word_count} | chars: {char_count})')
        else:
            print('\nNo sentences were generated. Try adjusting parameters')

        # Print out the statistics of the Markov generator
        generator.printStatistics()

        # Get the model info
        model_info = generator.getModelInfo()
        print(f'\n=== Model info ===\n\
- State size: {model_info.get("state_size", "N/A")},\n\
- Parsed sentences: {model_info.get("parsed_sentences", "N/A")},\n\
- Chain size: {model_info.get("chain_size", "N/A")}')
    except Exception as e:
        logger.error(f'Application error: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main()