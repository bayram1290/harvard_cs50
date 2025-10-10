import sys
from PIL import Image, ImageDraw, ImageFont
from crossword_creator import Crossword, Variable

class PasswordGenerator():

    def __init__(self, crossword) -> None:
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }


    def enforce_node_consistency(self) -> None:
        remove_words = {}

        for var, word_list in self.domains.items():
            for word in word_list:
                if len(word) != var.length:

                    if var not in remove_words:
                        remove_words[var] = []
                    remove_words[var].append(word)

        for var, word_list in remove_words.items():
            for word in word_list:
                self.domains[var].remove(word)


    def revise(self, var1, var2):
        overlap = self.crossword.overlaps[var1, var2]

        if overlap is None:
            return False

        index_x = overlap[0]
        index_y = overlap[1]

        remove_words = []
        for word in self.domains[var1]:
            possible = False
            for possible_word in self.domains[var2]:
                if word[index_x] == possible_word[index_y]:
                        possible = True

            if not possible:
                remove_words.append(word)

        if len(remove_words) == 0:
            return False

        for word in remove_words:
            self.domains[var1].remove(word)

        return True


    def ac3(self, arcs=None) -> bool:
        if arcs is None:

            arcs = []
            for var1 in self.domains:
                for var2 in self.crossword.neighbors(var1):
                    arcs.append((var1, var2))

        while len(arcs) != 0:
            (var1, var2) = arcs.pop(0)

            if self.revise(var1, var2):
                if len(self.domains[var1]) == 0:
                    return False

                var1_neighbor = self.crossword.neighbors(var1)
                var1_neighbor.remove(var2)

                for var in var1_neighbor:
                    arcs.append((var, var1))

        return True


    def assignment_complete(self, assignment) -> bool:
        return len(self.domains) == len(assignment)


    def select_unassigned_variable(self, assignment) -> Variable|None:
        candidate = None

        for v in self.domains:
            if v in assignment:
                continue

            if candidate is None or len(self.domains[v]) < len(self.domains[candidate]):
                candidate = v
            elif len(self.domains[v]) == len(self.domains[candidate]) and len(self.crossword.neighbors(v)) > len(self.crossword.neighbors(candidate)):
                candidate = v

        return candidate


    def order_domain_values(self, var: Variable, assignment: Crossword) -> dict:

        rule_out = {}
        neighbors = self.crossword.neighbors(var)

        for value in self.domains[var]:
            rule_out[value] = 0

            for other_var in self.domains:

                if other_var == var or other_var in assignment:
                    continue
                if value in self.domains[other_var]:
                    rule_out[value] += 1

            for neighbor in neighbors:
                if neighbor in assignment:
                    continue
                (index_var, index_neighbor) = self.crossword.overlaps[var, neighbor]

                for word in self.domains[neighbor]:
                    if value[index_var] != word[index_neighbor]:
                        rule_out[value] += 1

        return rule_out


    def consistent(self, assignment: dict) -> bool:

        value_set = set()
        for var, value in assignment.items():
            if value not in value_set:
                value_set.add(value)
            else:
                return False

            if var.length != len(value):
                return False

            for neighbor in self.crossword.neighbors(var):

                if neighbor in assignment:

                    (index_var, index_neighbor) = self.crossword.overlaps[var, neighbor]
                    if value[index_var] != assignment[neighbor][index_neighbor]:
                        return False

        return True


    def letter_grid(self, assignment: dict) -> list:
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]

        for var, word in assignment.items():
            direction = var.direction

            for k in range(len(word)):
                i = var.i + (k if direction == Variable.DOWN else 0)
                j = var.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]

        return letters

    def display(self, assignment: dict) -> None:
        letters = self.letter_grid(assignment)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                if self.crossword.structure[i][j]:
                    print(letters[i][j] or ' ', end='')
                else:
                    print("█", end='')

            print()


    def backtrack(self, assignment):

        if self.assignment_complete(assignment):
            return assignment

        var = self.select_unassigned_variable(assignment)
        if var is not None:
            for value in self.order_domain_values(var, assignment):

                assignment[var] = value
                if self.consistent(assignment):
                    self.ac3()
                    result = self.backtrack(assignment)

                    if result is not None:
                        return result

        return None


    def save_as_img(self, assignment: dict, savePath: str) -> None:

        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)
        height = self.crossword.height
        width = self.crossword.width
        img_extension = savePath[-3:]

        img = Image.new(
            mode=('RGBA' if img_extension == 'png' else 'RGB'),
            size=(width * cell_size, height * cell_size),
            color='black'
        )

        img_font = ImageFont.truetype('./fonts/OpenSans-Regular.ttf', 80)
        draw = ImageDraw.Draw(img)

        for i in range(height):
            for j in range(width):

                rect = [
                    (j * cell_size + cell_border, i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)
                ]

                if self.crossword.structure[i][j]:
                    draw.rectangle(xy=rect, fill='white')
                    if letters[i][j]:
                        bbox = draw.textbbox((0, 0), letters[i][j], font=img_font)
                        l_width, l_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

                        draw.text(
                            xy=(
                                (rect[0][0] + (interior_size - l_width) / 2),
                                (rect[0][1] + (interior_size - l_height) / 2 - 10)
                            ),
                            text=letters[i][j],
                            font=img_font,
                            fill='black'
                        )

        img.save(savePath)

    def solve(self):
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

def main():
    if len(sys.argv) not in [3, 4]:
        sys.exit('App usage: python app.py path_to_structure_file.txt path_to_words_file.txt [path_to_output_image.png/jpg/jpeg]')

    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # structure = './data/structure1.txt'
    # words = './data/words1.txt'
    # output = 'output.png'

    crossword = Crossword(structure, words)
    generator = PasswordGenerator(crossword)
    assignment = generator.solve()

    if assignment is None:
        print('No solution')
    else:
        generator.display(assignment)
        if output:
            generator.save_as_img(assignment, output)


if __name__ == '__main__':
    main()