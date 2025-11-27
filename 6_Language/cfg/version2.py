import nltk
import sys
from nltk.parse import ChartParser
from nltk.parse.generate import generate
from nltk import PCFG

class AdvancedCFGParser:
    def __init__(self):
        # Corrected probabilistic grammar with proper probability sums
        """
        Initialize the AdvancedCFGParser class.

        This class encapsulates both a standard Context-Free Grammar and
        a corrected probabilistic Context-Free Grammar, as well as
        parsing strategies for both grammars.

        Parameters:
        None

        Returns:
        None

        Attributes:
        parser (nltk.ChartParser): The parser for the standard CFG.
        cfg_parser (nltk.ChartParser): The parser for the probabilistic CFG.
        cfg (nltk.CFG): The probabilistic Context-Free Grammar.
        pcfg (nltk.PCFG): The corrected probabilistic Context-Free Grammar.
        """
        self.pcfg = PCFG.fromstring("""
            S -> NP VP [1.0]

            NP -> PropN [0.2] | N [0.3] | D NP [0.25] | AP NP [0.15] | NP PP [0.1]
            AP -> A [0.6] | A AP [0.4]
            PP -> P NP [1.0]
            VP -> V [0.2] | V NP [0.5] | V NP PP [0.3]

            # Fixed probabilities - each set sums to 1.0
            PropN -> "John" [0.3] | "Mary" [0.3] | "Bob" [0.2] | "Alice" [0.2]
            A -> "big" [0.2] | "blue" [0.2] | "small" [0.15] | "dry" [0.1] | "wide" [0.1] | "red" [0.1] | "happy" [0.15]
            D -> "the" [0.6] | "a" [0.3] | "an" [0.1]
            N -> "city" [0.15] | "car" [0.15] | "street" [0.1] | "dog" [0.15] | "binoculars" [0.1] | "man" [0.1] | "woman" [0.1] | "book" [0.15]
            P -> "on" [0.25] | "over" [0.2] | "before" [0.15] | "below" [0.15] | "with" [0.25]
            V -> "saw" [0.3] | "walked" [0.25] | "read" [0.2] | "runs" [0.25]
        """)

        # Standard CFG for comparison
        self.cfg = nltk.CFG.fromstring("""
            S -> NP VP
            NP -> PropN | N | D NP | AP NP | NP PP
            AP -> A | A AP
            PP -> P NP
            VP -> V | V NP | V NP PP

            PropN -> "John" | "Mary" | "Bob" | "Alice"
            A -> "big" | "blue" | "small" | "dry" | "wide" | "red" | "happy"
            D -> "the" | "a" | "an"
            N -> "city" | "car" | "street" | "dog" | "binoculars" | "man" | "woman" | "book"
            P -> "on" | "over" | "before" | "below" | "with"
            V -> "saw" | "walked" | "read" | "runs"
        """)

        self.parser = ChartParser(self.cfg)
        self.pcfg_parser = ChartParser(self.pcfg)

    def validate_probabilities(self):
        """
        Validates the probabilities of the probabilistic Context-Free Grammar (PCFG).

        This function checks that the probabilities of each left-hand side (LHS) in the PCFG sum up to 1.0.
        If any LHS has a sum of probabilities that is not close to 1.0, then the function prints an error message and returns False.
        Otherwise, it prints a success message and returns True.

        Returns
        -------
        bool
            True if all LHS probabilities sum up to 1.0, False otherwise.
        """
        print("🔍 Validating PCFG probabilities:")

        lhs_prob_sums = {}
        for production in self.pcfg.productions():
            lhs = production.lhs()
            prob = production.prob()

            if lhs not in lhs_prob_sums:
                lhs_prob_sums[lhs] = 0.0
            lhs_prob_sums[lhs] += prob

        all_valid = True
        for lhs, total_prob in lhs_prob_sums.items():
            status = "✅" if abs(total_prob - 1.0) < 1e-10 else "❌"
            print(f"  {status} {lhs}: {total_prob:.2f}")
            if abs(total_prob - 1.0) >= 1e-10:
                all_valid = False

        return all_valid

    def parse_sentence(self, sentence):
        """
        Parse a sentence using the Context-Free Grammar.

        Parameters
        ----------
        sentence : str
            The sentence to parse.

        Returns
        -------
        list of nltk.Tree
            A list of parsed trees if the parse was successful, None otherwise.

        Notes
        -----
        This function will display the parse trees and metrics for each tree.
        If no parse trees were found, it will print a message indicating this.
        """
        tokens = sentence.split()
        results = []

        try:
            # Get all parse trees
            trees = list(self.parser.parse(tokens))

            if not trees:
                print("❌ No valid parse trees found.")
                return []

            print(f"🌳 Found {len(trees)} possible parse tree(s):")

            for i, tree in enumerate(trees, 1):
                print(f"\n📊 Tree {i}:")
                tree.pretty_print()
                print(f"Tree height: {tree.height()}")
                print(f"Number of leaves: {len(tree.leaves())}")

                # Extract phrases
                self._extract_phrases(tree)
                results.append(tree)

            return results

        except ValueError as e:
            print(f"❌ Parsing error: {e}")
            return []
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return []

    def probabilistic_parse(self, sentence):
        """
        Parse a sentence using the probabilistic Context-Free Grammar (PCFG).

        Parameters
        ----------
        sentence : str
            The sentence to parse.

        Returns
        -------
        nltk.Tree or None
            The most likely parse tree if the parse was successful, None otherwise.

        Notes
        -----
        This function will display the most likely parse tree and its probability.
        If no parse trees were found, it will print a message indicating this.
        """
        tokens = sentence.split()

        try:
            trees = list(self.pcfg_parser.parse(tokens))
            if trees:
                print("🎯 Probabilistic parse (most likely):")
                trees[0].pretty_print()

                # Calculate and show tree probability
                tree_prob = self._calculate_tree_probability(trees[0])
                print(f"Tree probability: {tree_prob:.6f}")

                return trees[0]
            return None
        except Exception as e:
            print(f"Probabilistic parsing failed: {e}")
            return None

    def _calculate_tree_probability(self, tree):
        """
        Calculate the probability of a parse tree.

        Parameters
        ----------
        tree : nltk.Tree
            The parse tree to calculate the probability of.

        Returns
        -------
        float
            The probability of the parse tree.

        Notes
        -----
        This function multiplies the probabilities of all productions in the tree.
        """
        probability = 1.0
        for production in tree.productions():
            probability *= production.prob()
        return probability

    def _extract_phrases(self, tree):
        """
        Extract phrases from a parse tree.

        Parameters
        ----------
        tree : nltk.Tree
            The parse tree to extract phrases from.

        Notes
        -----
        This function will extract phrases of type NP, VP, PP, and AP from the parse tree and print them out.
        """
        phrases = {
            'NP': [],
            'VP': [],
            'PP': [],
            'AP': []
        }

        for subtree in tree.subtrees():
            if subtree.label() in phrases:
                phrases[subtree.label()].append(' '.join(subtree.leaves()))

        print("\n🔍 Extracted phrases:")
        for phrase_type, instances in phrases.items():
            if instances:
                print(f"  {phrase_type}: {instances}")

    def generate_sample_sentences(self, num=5):
        """
        Generate sample sentences from the grammar.

        Parameters
        ----------
        num : int, optional
            The number of sample sentences to generate. Defaults to 5.

        Notes
        -----
        This function will generate sample sentences from the grammar and print them out.
        The sentences will be generated with a depth of 7.
        """
        print(f"\n💡 Sample sentences generated from grammar:")
        for i, sent in enumerate(generate(self.cfg, depth=7), 1):
            if i > num:
                break
            print(f"  {i}. {' '.join(sent)}")

    def validate_sentence_structure(self, sentence):
        """
        Validate if a sentence has a valid structure in our grammar.

        A sentence is considered to have a valid structure if it contains at least one noun and one verb.
        The function prints out the tokens, if the sentence contains a noun, if the sentence contains a verb, and if the sentence has a valid structure.

        Parameters
        ----------
        sentence : str
            The sentence to validate.

        Returns
        -------
        None

        Notes
        -----
        This function is mainly used for debugging purposes.
        """
        tokens = sentence.split()
        has_noun = any(self._is_noun(token) for token in tokens)
        has_verb = any(self._is_verb(token) for token in tokens)

        print(f"\n🔎 Sentence analysis:")
        print(f"  Tokens: {tokens}")
        print(f"  Has noun: {has_noun}")
        print(f"  Has verb: {has_verb}")
        print(f"  Valid structure: {has_noun and has_verb}")

    def _is_noun(self, word):
        """
        Check if a word can be a noun in our grammar.

        Parameters
        ----------
        word : str
            The word to check.

        Returns
        -------
        bool
            True if the word can be a noun, False otherwise.
        """
        noun_productions = [prod for prod in self.cfg.productions() if prod.lhs().symbol() == 'N']
        return any(word in prod.rhs() for prod in noun_productions)

    def _is_verb(self, word):
        """
        Check if a word can be a verb in our grammar.

        Parameters
        ----------
        word : str
            The word to check.

        Returns
        -------
        bool
            True if the word can be a verb, False otherwise.
        """
        verb_productions = [prod for prod in self.cfg.productions() if prod.lhs().symbol() == 'V']
        return any(word in prod.rhs() for prod in verb_productions)

    def interactive_mode(self):
        """
        Enter interactive mode with the Advanced CFG Parser.

        In this mode, you can enter sentences to parse, generate sample sentences, show the vocabulary, validate PCFG probabilities, or quit the program.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Notes
        -----
        This function is the main entry point for the interactive mode of the Advanced CFG Parser.
        """
        print("🚀 Advanced CFG Parser - Interactive Mode")
        print("Available commands:")
        print("  - Enter a sentence to parse it")
        print("  - 'samples' - show sample sentences")
        print("  - 'vocab' - show vocabulary")
        print("  - 'validate' - validate PCFG probabilities")
        print("  - 'quit' - exit the program")
        print("-" * 50)

        # Validate probabilities on startup
        self.validate_probabilities()

        while True:
            try:
                user_input = input("\n📝 Enter sentence or command: ").strip()

                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'samples':
                    self.generate_sample_sentences()
                elif user_input.lower() == 'vocab':
                    self.show_vocabulary()
                elif user_input.lower() == 'validate':
                    self.validate_probabilities()
                elif user_input:
                    self.validate_sentence_structure(user_input)
                    self.parse_sentence(user_input)
                    self.probabilistic_parse(user_input)

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def show_vocabulary(self):
        """
        Print out the vocabulary of the grammar, categorized by part of speech.

        Prints out the vocabulary of the grammar, categorized by part of speech.
        The categories are 'Nouns', 'Verbs', 'Adjectives', 'Determiners', 'Prepositions', and 'Proper Nouns'.
        """
        vocab = {
            'Nouns': set(), 'Verbs': set(), 'Adjectives': set(), 
            'Determiners': set(), 'Prepositions': set(), 'Proper Nouns': set()
        }

        for prod in self.cfg.productions():
            if prod.is_lexical():
                word = str(prod.rhs()[0])
                lhs = prod.lhs().symbol()
                if lhs == 'N': vocab['Nouns'].add(word)
                elif lhs == 'V': vocab['Verbs'].add(word)
                elif lhs == 'A': vocab['Adjectives'].add(word)
                elif lhs == 'D': vocab['Determiners'].add(word)
                elif lhs == 'P': vocab['Prepositions'].add(word)
                elif lhs == 'PropN': vocab['Proper Nouns'].add(word)

        print("\n📚 Vocabulary:")
        for category, words in vocab.items():
            if words:
                print(f"  {category}: {', '.join(sorted(words))}")

def main():
    """
    Main function to run the program.

    If command line arguments are provided, it will parse the given sentence.
    Otherwise, it will run in interactive mode.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    parser = AdvancedCFGParser()

    if len(sys.argv) > 1:
        # Command line mode
        sentence = ' '.join(sys.argv[1:])
        parser.parse_sentence(sentence)
    else:
        # Interactive mode
        parser.interactive_mode()

if __name__ == "__main__":
    main()