class Nim():
    def __init__(self) -> None:
        """
        Initialize a Nim game object.

        The game object has three attributes:
        - `piles`: a list of how many elements remain in each pile
        - `player`: 0 or 1 to indicate which player's turn
        - `winner`: None, 0, or 1 to indicate who the winner is
        """
        initial = [1,3,5,7]
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_acts(cls, piles):
        """
        Given a list of piles, return all possible actions that can be taken
        from that state.

        An action is a tuple `(i, j)` where `i` is the pile index and `j` is the
        number of items to remove from that pile.

        Returns a set of all possible actions.
        """
        acts = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                acts.add((i, j))
        return acts

    @classmethod
    def other_player(cls, player):
        return 0 if player == 1 else 1

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, act):
        """
        Make the move `act` for the current player.

        `act` must be a tuple `(i, j)` where `i` is the pile index and
        `j` is the number of items to remove from that pile.

        Raises an exception if the game has already been won or if the
        action is invalid.

        Updates the game state by removing items from the pile and
        switching the current player.

        If all piles are empty after the move, sets the winner to the current
        player.
        """
        pile, count = act

        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        self.piles[pile] -= count
        self.switch_player()

        if all(pile == 0 for pile in self.piles):
            self.winner = self.player
