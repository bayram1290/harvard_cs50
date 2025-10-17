from nim import Nim
from nimAI import Nimmer
import random
import time


def modelTrain(epoch: int)->Nimmer:
    """
    Train a Nim AI model for a given number of epochs.

    Args:
        epoch (int): The number of training games to play.

    Returns:
        Nimmer: The trained Nim AI model.
    """
    player = Nimmer()
    for i in range(epoch):
        print(f'Playing training game: {i+1}')
        game = Nim()
        tracker={
            0: {'state': None, 'act': None},
            1: {'state': None, 'act': None},
        }
        while True:
            state = game.piles.copy()
            act = player.act(game.piles)

            tracker[game.player]['state'] = state
            tracker[game.player]['act'] = act

            game.move(act)
            newState = game.piles.copy()

            if game.winner is not None:
                player.updateModel(state, act, newState, -1)
                player.updateModel(tracker[game.player]['state'], tracker[game.player]['act'], newState, 1)
                break
            elif tracker[game.player]['state'] is not None:
                player.updateModel(tracker[game.player]['state'], tracker[game.player]['act'], newState, 0)

    print('Training is complete')
    return player

def playGame(model: Nimmer, humanPlayer: int|None=None):
    """
    Play a game of Nim against the AI model.

    Args:
        model (Nimmer): The trained Nim AI model.
        humanPlayer (int|None): The player to make the first move. If None, the starting player is chosen randomly.

    Returns:
        None
    """
    if humanPlayer == None:
        humanPlayer = random.randint(0, 1)

    game = Nim()

    while True:
        print('\nPiles:')
        for i, pile in enumerate(game.piles):
            print(f'Pile {i}: {pile}')

        available_acts = Nim.available_acts(game.piles)
        time.sleep(1)

        if game.player == humanPlayer:
            print('Your turn')
            while True:
                pile = int(input('Choose pile: '))
                count = int(input('Enter count: '))

                if (pile, count) in available_acts:
                    break
                print('Invalid move, try again')
        else:
            print("Computer's turn")
            pile, count = model.act(state=game.piles, epsilon=False)
            print(f"Computer turn is: Get {count} from pile {pile}.")

        game.move((pile, count))
        if game.winner is not None:
            print('\nGame is over')
            winner = 'You' if game.winner == humanPlayer else 'Computer'
            print(f'Winner is {winner}')

            return


if __name__ == '__main__':
    model = modelTrain(1000)
    playGame(model)
