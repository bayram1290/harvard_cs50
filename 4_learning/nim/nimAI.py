from nim import Nim
import random
import math

class Nimmer():
    def __init__(self, alpha: float=0.5, epsilon: float=0.1) -> None:
        """
        Initialize Nimmer AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)` pairs to a Q-value (a number).
        - `state` is a tuple of remaining piles, e.g. (1, 1, 4, 4)
        - `action` is a tuple `(i, j)` for an action
        """

        self.Q = dict()
        self.alpha = alpha
        self.epsilon = epsilon


    def _getQ(self, state, act):
        """
        Return the Q-value for the state `state` and the action `act`.
        If no Q-value exists yet in `self.Q`, return 0.
        """
        if (tuple(state), act) in self.Q:
            return self.Q[(tuple(state), act)]
        return 0

    def _updateQ(self, state, act, oldQ, reward, value) -> None:
        """
        Update the Q-value for the state `state` and the action `act`
        given the previous Q-value `oldQ`, a current reward `reward`,
        and an estimate of future rewards `value`.

        The Q-value is updated according to the formula:
        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        newValue = reward + value
        self.Q[(tuple(state), act)] = oldQ + self.alpha * (newValue - oldQ)

    def _getValue(self, state):
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a `(state, action)` pair has no
        Q-value in `self.Q`. If there are no available actions in
        `state`, return 0.
        """
        possible_acts = Nim.available_acts(state)

        if len(possible_acts) == 0:
            return 0

        value = -math.inf
        for act in possible_acts:
            currentValue = self._getQ(state, act)
            if currentValue > value:
                value = currentValue

        return value

    def act(self, state, epsilon: bool=True):
        """
        Choose an action given the current state of the game.

        If `epsilon` is `True`, then choose a random available action
        according to the probabilities given by the weights.

        Otherwise, choose the best action available according to
        the Q-values.

        Returns:
            action: The chosen action as a tuple `(i, j)`.
        """
        possible_actions = Nim.available_acts(state)
        highest_q = -math.inf
        best_action = None

        for action in possible_actions:
            current_q = self._getQ(state, action)
            if current_q > highest_q:
                highest_q = current_q
                best_action = action

        if epsilon:
            action_weights = [self.epsilon / len(possible_actions) if action != best_action else
                                (1 - self.epsilon) for action in possible_actions]

            best_action = random.choices(list(possible_actions), weights=action_weights, k=1)[0]

        return best_action

    def updateModel(self, oldState, act, newState, reward):
        """
        Update the Q-learning model given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.

        Parameters:
            oldState (tuple): The old state of the game.
            act (tuple): The action taken in the old state.
            newState (tuple): The new resulting state of the game.
            reward (int): The reward received from taking the action.
        """
        oldQ = self._getQ(oldState, act)
        value = self._getValue(newState)
        self._updateQ(oldState, act, oldQ, reward, value)
