import random
from PIL import Image, ImageDraw, ImageFont

img_house = Image.open('./media/img/house.png')
img_hospital = Image.open('./media/img/hospital.png')
img_font = ImageFont.truetype('./media/fonts/OpenSans-Regular.ttf', 32)

class Space:
    """
    A class to represent a space for placing hospitals.

    The space is a grid of a certain height and width, with a certain number of hospitals to place.
    """

    def __init__(self, height:int, width:int, hospital_count:int, debug:bool):
        """
        Initialize a Space object.

        Parameters:
        height (int): The height of the space.
        width (int): The width of the space.
        hospital_count (int): The number of hospitals to place.
        debug (bool): Whether to print debug messages.
        """
        self.h = height
        self.w = width
        self.hospital_cnt = hospital_count
        self.houses = set()
        self.hospitals = set()
        self.debug = debug

    def add_houses(self, house_count:int):
        """
        Add houses to the space.

        Parameters:
        house_count (int): The number of houses to add.
        """
        for _ in range(house_count):

            r = random.randrange(self.h)
            c = random.randrange(self.w)

            while (r, c) in self.houses:
                r = random.randrange(self.h)
                c = random.randrange(self.w)

            self.houses.add((r, c))

    def _get_available_space(self):
        """
        Get the available space in the grid.

        Returns:
        set: A set of available space coordinates.
        """
        candidates = set(
            (row, col)
            for row in range(self.h)
            for col in range(self.w)
        )

        for house in self.houses:
            candidates.remove(house)

        for hospital in self.hospitals:
            candidates.remove(hospital)

        return candidates

    def _get_cost(self, hospitals):
        """
        Calculate the cost of the current state.

        Parameters:
        hospitals (set): A set of hospital coordinates.

        Returns:
        int: The cost of the current state.
        """
        cost = 0

        for house in self.houses:
            cost += min(
                abs(hospital[0] - house[0]) + abs(hospital[1] - house[1]) for hospital in hospitals
            )

        return cost

    def _draw_placement(self, image_name:str):
        """
        Draw the current state of the space.

        Parameters:
        image_name (str): The name of the image to save.
        """
        cell_size = 100
        border_size = 2
        padding=10
        cost_size = 40

        house = img_house.resize((cell_size ,cell_size))
        hospital = img_hospital.resize((cell_size ,cell_size))

        img = Image.new(
            mode='RGBA',
            size=(self.w * cell_size, self.h * cell_size + cost_size + padding * 2),
            color='white'
        )

        draw = ImageDraw.Draw(img)

        for row_cell in range(self.h):
            for col_cell in range(self.w):

                cell_rect = [
                    (col_cell * cell_size + border_size, row_cell * cell_size + border_size),
                    ((col_cell + 1) * cell_size - border_size, (row_cell + 1) * cell_size - border_size)
                ]
                draw.rectangle(xy=cell_rect, fill='black')

                if (row_cell, col_cell) in self.hospitals:
                    img.paste(im=hospital, box=cell_rect[0], mask=hospital)

                if (row_cell, col_cell) in self.houses:
                    img.paste(im=house, box=cell_rect[0], mask=house)

        draw.rectangle(
            xy=(0, self.h * cell_size + padding, self.w * cell_size, self.h * cell_size + padding*2 + cost_size),
            fill='blue'
        )

        draw.text(
            (padding, self.h * cell_size + padding),
            text=f'Cost: {self._get_cost(self.hospitals)}',
            font=img_font,
            fill='white',
            stroke_width=0.35
        )

        img.save(image_name)

    def _get_neighbors(self, row, col):
        """
        Get the neighbors of a given cell.

        Parameters:
        row (int): The row of the cell.
        col (int): The column of the cell.

        Returns:
        list: A list of neighbor coordinates.
        """
        candidates = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        neighbors = []

        for r, c in candidates:
            if (r, c) in self.hospitals or (r, c) in self.houses:
                continue
            if 0 <= r < self.h and 0 <= c < self.w:
                neighbors.append((r, c))

        return neighbors

    def hill_climb(self, image_prefix:str|None=None, max_process_cnt:int|None=None):
        """
        Perform a hill climb algorithm to find the best placement of the hospitals.

        Parameters:
        image_prefix (str|None): The prefix for the image names.
        max_process_cnt (int|None): The maximum number of iterations.

        Returns:
        set: A set of the best hospital coordinates.
        int: The cost of the best state.
        """
        process_count = 0
        self.hospitals = set()
        for _ in range(self.hospital_cnt):
            self.hospitals.add(random.choice(list(self._get_available_space())))

        initial_cost = self._get_cost(self.hospitals)
        if self.debug:
            print(f'Initial state cost: {initial_cost}')

        if image_prefix:
            self._draw_placement(image_name=f'{image_prefix}{str(process_count).zfill(3)}.png')

        while max_process_cnt is None or process_count < max_process_cnt:
            process_count += 1
            best_neighbors = []
            best_neighbor_cost = None

            for hospital in self.hospitals:
                for replacement in self._get_neighbors(*hospital):

                    neighbor = self.hospitals.copy()
                    neighbor.remove(hospital)
                    neighbor.add(replacement)

                    cost = self._get_cost(neighbor)

                    if best_neighbor_cost is None or cost < best_neighbor_cost:
                        best_neighbor_cost = cost
                        best_neighbors = [neighbor]
                    elif cost == best_neighbor_cost:
                        best_neighbors.append(neighbor)

            if best_neighbor_cost is not None:
                if best_neighbor_cost >= self._get_cost(self.hospitals):
                    return self.hospitals
                else:
                    if self.debug:
                        print(f'New better cost: {best_neighbor_cost}')
                    self.hospitals = random.choice(best_neighbors)

                    if image_prefix:
                        self._draw_placement(image_name=f'{image_prefix}{str(process_count).zfill(3)}.png')

    def random_restart(self, max_process_cnt:int, image_prefix:str|None):
        """
        Perform a random restart algorithm to find the best placement of the hospitals.

        Parameters:
        max_process_cnt (int): The maximum number of iterations.
        image_prefix (str|None): The prefix for the image names.

        Returns:
        set: A set of the best hospital coordinates.
        int: The cost of the best state.
        """
        best_hospitals = None
        best_cost = None

        for i in range(max_process_cnt):
            hospitals = self.hill_climb()
            cost = self._get_cost(hospitals)
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_hospitals = hospitals

                if self.debug:
                    print(f'{i + 1}: Found new best state cost: {cost}')
            else:
                if self.debug:
                    print(f'{i + 1}: Found state cost: {cost}')

            if image_prefix:
                self._draw_placement(image_name=f'{image_prefix + '_'}.{str(i).zfill(3)}.png')

        return best_hospitals, best_cost

def main():
    """
    Main function to test the Space class.

    This function creates a Space object with a 10x20 grid, 3 hospitals, and debug mode enabled.
    It then adds 15 houses to the grid and runs the random_restart method to find the best placement of the hospitals.
    The resulting hospitals and score are not used.
    """
    s = Space(height=10, width=20, hospital_count=3, debug=True)
    s.add_houses(house_count=15)

    # __ = s.hill_climb(image_prefix='hospitals')
    hospitals, score = s.random_restart(max_process_cnt=20, image_prefix='hospitals')

if __name__ == "__main__":
    main()