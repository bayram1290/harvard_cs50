import nltk
from nltk.parse.generate import generate
import random
from collections import Counter

class GrammarParser:
    def __init__(self):
        # Enhanced grammar with more variety
        self.grammar = nltk.CFG.fromstring("""
            S -> NP_SG VP_SG | NP_PL VP_PL | S CONJ S | IMPERATIVE
            IMPERATIVE -> V_IMP | V_IMP NP
            
            NP_SG -> D_SG N_SG | D_SG ADJ N_SG | NP_SG PP | PRON_SG | PROP_SG
            NP_PL -> D_PL N_PL | D_PL ADJ N_PL | NP_PL PP | PRON_PL | PROP_PL
            
            VP_SG -> V_SG_INTRANS | V_SG_TRANS NP | V_SG_LINK ADJ | VP_SG PP | VP_SG ADV | AUX VP_SG
            VP_PL -> V_PL_INTRANS | V_PL_TRANS NP | V_PL_LINK ADJ | VP_PL PP | VP_PL ADV | AUX VP_PL
            
            PP -> P NP
            
            # Expanded vocabulary for more variety
            D_SG -> "the" | "a" | "an" | "this" | "that" | "my" | "your"
            D_PL -> "the" | "these" | "those" | "my" | "your" | "many"
            
            # More nouns with semantic categories
            N_SG -> "book" | "car" | "house" | "city" | "computer" | "problem" | "solution" | "idea"
            N_PL -> "books" | "cars" | "houses" | "cities" | "computers" | "problems" | "solutions" | "ideas"
            N_ANIMATE_SG -> "student" | "teacher" | "doctor" | "artist" | "child" | "dog" | "cat" | "friend"
            N_ANIMATE_PL -> "students" | "teachers" | "doctors" | "artists" | "children" | "dogs" | "cats" | "friends"
            
            # More verbs with semantic constraints
            V_SG_INTRANS -> "works" | "sleeps" | "exists" | "appears" | "arrives" | "disappears"
            V_PL_INTRANS -> "work" | "sleep" | "exist" | "appear" | "arrive" | "disappear"
            
            V_SG_TRANS -> "reads" | "writes" | "builds" | "solves" | "explains" | "understands" | "creates"
            V_PL_TRANS -> "read" | "write" | "build" | "solve" | "explain" | "understand" | "create"
            
            V_SG_LINK -> "is" | "seems" | "becomes" | "looks" | "appears" | "feels"
            V_PL_LINK -> "are" | "seem" | "become" | "look" | "appear" | "feel"
            
            V_IMP -> "read" | "write" | "build" | "solve" | "listen" | "think"
            
            # Expanded pronouns and proper nouns
            PRON_SG -> "I" | "you" | "he" | "she" | "it"
            PRON_PL -> "we" | "they" | "you"
            PROP_SG -> "John" | "Mary" | "Alice" | "Bob" | "Emma" | "David" | "Sarah" | "Michael"
            
            ADJ -> "interesting" | "difficult" | "beautiful" | "important" | "simple" | "complex" | "new" | "old"
            ADV -> "carefully" | "quickly" | "easily" | "slowly" | "happily" | "clearly" | "well"
            P -> "in" | "on" | "at" | "with" | "about" | "for" | "through" | "during"
            CONJ -> "and" | "or" | "but" | "while" | "because" | "although"
            AUX -> "can" | "will" | "should" | "might" | "must" | "could"
        """)
        
        # Multiple parsing strategies
        self.parsers = {
            'chart': nltk.ChartParser(self.grammar),
            'recursive_descent': nltk.RecursiveDescentParser(self.grammar),
        }
        
        # Statistics tracking
        self.stats = Counter()
        
    def parse_sentence(self, sentence, parser_type='chart'):
        """Parse sentence with chosen parser type"""
        tokens = sentence.split()
        self.stats['total_parses'] += 1
        
        try:
            parser = self.parsers[parser_type]
            trees = list(parser.parse(tokens))
            
            if trees:
                self.stats['successful_parses'] += 1
                return trees
            else:
                return None
                
        except ValueError as e:
            print(f"Parsing error: {e}")
            return None
    
    def display_results(self, trees, sentence):
        """Display parsing results in multiple formats"""
        if not trees:
            print(f"\n❌ No parse trees possible for: '{sentence}'")
            return
        
        print(f"\n✅ Found {len(trees)} parse tree(s) for: '{sentence}'")
        print("=" * 60)
        
        for i, tree in enumerate(trees, 1):
            print(f"\n🌳 Tree {i}:")
            tree.pretty_print()
            
            # Calculate and display tree metrics
            height = tree.height()
            leaves = len(tree.leaves())
            print(f"📊 Metrics - Height: {height}, Leaves: {leaves}")
            
        # Ask user if they want to visualize
        self.ask_visualization(trees[0])
    
    def ask_visualization(self, tree):
        """Ask user if they want to visualize the tree"""
        response = input("\n🎨 Visualize tree? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            try:
                tree.draw()
                print("🖼️ Tree visualization opened in new window")
            except Exception as e:
                print(f"⚠️ Visualization error: {e}")
    
    def generate_examples(self, count=10):
        """Dynamically generate example sentences from the grammar"""
        print(f"\n📚 Generating {count} example sentences:")
        print("-" * 40)
        
        generated_count = 0
        attempts = 0
        max_attempts = count * 10
        
        try:
            for sentence in generate(self.grammar, depth=6):
                sent_str = ' '.join(sentence)
                
                if self._is_sensible_sentence(sent_str) and len(sent_str.split()) >= 3:
                    print(f"{generated_count + 1}. {sent_str}")
                    generated_count += 1
                    attempts = 0
                
                attempts += 1
                if generated_count >= count or attempts >= max_attempts:
                    break
                    
        except RuntimeError as e:
            print(f"⚠️  Generation stopped: {e}")
        
        if generated_count < count:
            print(f"\nℹ️  Only generated {generated_count} examples")
            
        return generated_count
    
    def _is_sensible_sentence(self, sentence):
        """Basic filter for semantically sensible sentences"""
        words = sentence.lower().split()
        
        if len(words) < 2:
            return False
            
        nonsensical_patterns = [
            ("the", "city", "walks"), ("the", "book", "runs"), 
            ("the", "house", "eats"), ("the", "computer", "sleeps"),
            ("a", "city", "reads"), ("the", "problem", "walks")
        ]
        
        for pattern in nonsensical_patterns:
            if all(word in sentence.lower() for word in pattern):
                return False
                
        return True
    
    def generate_diverse_examples(self, count=15):
        """Generate diverse examples using template-based approach"""
        print(f"\n🎲 Generating {count} DIVERSE examples:")
        print("-" * 45)
        
        examples = set()
        
        # Word banks with actual words from our grammar
        word_banks = {
            'adj': ['interesting', 'difficult', 'beautiful', 'important', 'simple', 'complex', 'new'],
            'n_sg': ['book', 'problem', 'solution', 'idea', 'city', 'house', 'computer'],
            'n_pl': ['books', 'problems', 'solutions', 'ideas', 'cities', 'houses', 'computers'],
            'n_animate_sg': ['student', 'teacher', 'artist', 'doctor', 'child', 'dog'],
            'n_animate_pl': ['students', 'teachers', 'artists', 'doctors', 'children', 'dogs'],
            'v_sg_trans': ['reads', 'writes', 'solves', 'explains', 'understands', 'creates'],
            'v_sg_intrans': ['works', 'sleeps', 'exists', 'appears', 'arrives'],
            'v_pl_trans': ['read', 'write', 'solve', 'explain', 'understand', 'create'],
            'v_pl_intrans': ['work', 'sleep', 'exist', 'appear', 'arrive'],
            'v_sg_link': ['is', 'seems', 'becomes', 'looks', 'appears'],
            'v_pl_link': ['are', 'seem', 'become', 'look', 'appear'],
            'adv': ['carefully', 'quickly', 'easily', 'slowly', 'happily', 'well'],
            'prop_sg': ['John', 'Mary', 'Alice', 'Bob', 'Emma', 'David'],
            'pron_sg': ['he', 'she', 'it'],
            'pron_pl': ['they', 'we'],
            'd_sg': ['the', 'a', 'this', 'that'],
            'd_pl': ['the', 'these', 'those'],
            'p': ['in', 'on', 'with', 'about', 'for']
        }
        
        # Better templates that match our grammar structure
        templates = [
            # Simple sentences
            "{d_sg} {n_animate_sg} {v_sg_trans} {d_sg} {n_sg}",
            "{prop_sg} {v_sg_intrans} {adv}",
            "{d_sg} {adj} {n_sg} {v_sg_link} {adj}",
            "{pron_pl} {v_pl_trans} {d_sg} {n_sg}",
            
            # Slightly more complex
            "{d_sg} {n_animate_sg} {v_sg_trans} {d_sg} {n_sg} {p} {d_sg} {n_sg}",
            "{prop_sg} {v_sg_trans} {d_sg} {adj} {n_sg}",
            "{d_sg} {n_animate_pl} {v_pl_intrans} {adv}",
            "{pron_sg} {v_sg_link} {adj}",
            
            # With different structures
            "{d_sg} {n_sg} {v_sg_link} {adj}",
            "{prop_sg} and {prop_sg} {v_pl_trans} {d_sg} {n_pl}",
            "{d_sg} {n_animate_sg} {v_sg_trans} {adv}",
            "{pron_pl} {v_pl_link} {adj}"
        ]
        
        attempts = 0
        max_attempts = count * 5
        
        while len(examples) < count and attempts < max_attempts:
            # Pick a random template
            template = random.choice(templates)
            sentence = template
            
            # Replace all placeholders with actual words
            try:
                for placeholder, word_list in word_banks.items():
                    while f"{{{placeholder}}}" in sentence:
                        sentence = sentence.replace(f"{{{placeholder}}}", random.choice(word_list), 1)
                
                # Check if the sentence parses successfully
                if self._sentence_parses(sentence):
                    examples.add(sentence)
                    
            except Exception as e:
                # If substitution fails, just continue
                pass
                
            attempts += 1
        
        # Display the generated examples
        final_examples = list(examples)[:count]
        for i, example in enumerate(final_examples, 1):
            print(f"{i}. {example}")
            
        if len(final_examples) < count:
            print(f"\nℹ️  Generated {len(final_examples)} out of {count} requested examples")
            
        return len(final_examples)
    
    def _sentence_parses(self, sentence):
        """Check if a sentence can be parsed by our grammar (silent version)"""
        tokens = sentence.split()
        try:
            parser = self.parsers['chart']
            trees = list(parser.parse(tokens))
            return len(trees) > 0
        except:
            return False
    
    def grammar_coverage(self, sentence):
        """Analyze grammar coverage for a sentence"""
        tokens = sentence.split()
        covered = []
        missing = []
        
        for token in tokens:
            if any(token in prod.rhs() for prod in self.grammar.productions()):
                covered.append(token)
            else:
                missing.append(token)
        
        coverage = len(covered) / len(tokens) * 100
        return coverage, covered, missing
    
    def show_stats(self):
        """Display parsing statistics"""
        print(f"\n📈 Parser Statistics:")
        print(f"   Total parse attempts: {self.stats['total_parses']}")
        print(f"   Successful parses: {self.stats['successful_parses']}")
        if self.stats['total_parses'] > 0:
            success_rate = (self.stats['successful_parses'] / self.stats['total_parses']) * 100
            print(f"   Success rate: {success_rate:.1f}%")
    
    def interactive_mode(self):
        """Run interactive parsing session"""
        print("🚀 Advanced Grammar Parser (Fixed Template Generation)")
        print("=" * 60)
        print("📝 Template generation now works correctly!")
        
        while True:
            print(f"\nOptions:")
            print("1. Parse a sentence")
            print("2. Generate basic examples") 
            print("3. Generate DIVERSE examples (recommended)")
            print("4. Show statistics")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                sentence = input("\nEnter sentence to parse: ").strip()
                if not sentence:
                    continue
                
                # Check grammar coverage
                coverage, covered, missing = self.grammar_coverage(sentence)
                print(f"\n📖 Grammar coverage: {coverage:.1f}%")
                if missing:
                    print(f"⚠️  Unknown words: {', '.join(missing)}")

                # Parse with multiple strategies
                for parser_name in self.parsers.keys():
                    print(f"\n🔍 Trying {parser_name.replace('_', ' ').title()} parser:")
                    trees = self.parse_sentence(sentence, parser_name)
                    self.display_results(trees, sentence)
                    if trees:
                        break

            elif choice == '2':
                count = input("How many examples? (default 10): ").strip()
                count = int(count) if count.isdigit() else 10
                self.generate_examples(count)

            elif choice == '3':
                count = input("How many diverse examples? (default 15): ").strip()
                count = int(count) if count.isdigit() else 15
                self.generate_diverse_examples(count)

            elif choice == '4':
                self.show_stats()

            elif choice == '5':
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice. Please try again.")

def main():
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    # Create and run parser
    parser = GrammarParser()
    parser.interactive_mode()

if __name__ == "__main__":
    main()