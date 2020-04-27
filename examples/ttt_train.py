import argparse

from apprentice.agents.Memo import Memo
from apprentice.agents.RLAgent import RLAgent
from apprentice.agents.soartech_agent import SoarTechAgent
from apprentice.working_memory import ExpertaWorkingMemory
from ttt_experta import TicTacToe, TTTEngineBasic
from ttt_solver import best_move


def play_game_manually(human_role='O'):
    game = TicTacToe()

    new_game = True
    while new_game:
        game.reset()

        while game.winner() is None:
            print()
            print("Current Player: " + game.current_player)
            print(game)
            if game.current_player == human_role:

                try:
                    loc = input("Enter move as row and column integers (e.g."
                                ", 1,2):")
                    loc = loc.split(',')

                    row = int(loc[0])
                    col = int(loc[1])
                    game.mark(row, col, game.current_player)
                except ValueError:
                    print("############################")
                    print("# Invalid move, try again. #")
                    print("############################")
            else:

                r, c, v = best_move(game.list_rep())
                game.mark(r, c, v)

        if game.winner() == "DRAW":
            print("DRAW")
        else:
            print(game.winner() + " WINS!")

        new = input("Play again? Press enter to continue or type 'no' to"
                    " stop.")
        new_game = new == ""


def train_agent(agent, agent_role='O', use_oracle=True, max_games=10):
    game = TicTacToe()

    new_game = True
    game_count = 0
    while new_game:
        game_count += 1
        game.reset()

        while game.winner() is None:
            # print(str(game))
            # time.sleep(.3)

            if game.current_player.value == agent_role:
                original_state = game.machine_rep()

                try:
                    agent_action = agent.request(original_state)
                    if agent_action == {}:
                        # print("requesting assistance")
                        if use_oracle:
                            row, col, player = best_move(game.list_rep())
                            assert player == game.current_player.value
                        else:
                            print(
                                "Agent unsure what to do, enter move to "
                                "demonstrate"
                                " correct behavior")

                            loc = input(
                                "Enter move as row and column integers (e.g."
                                ", 1,2):")
                            loc = loc.split(',')

                            row = int(loc[0])
                            col = int(loc[1])
                            player = game.current_player

                        current_sai = game.mark(row, col, player)

                        correctness = True

                    else:

                        row, col = agent_action.selection
                        player = agent_action.input['value']
                        current_sai = agent_action

                        if use_oracle:
                            correctness = (row, col, player) == best_move(
                                game.list_rep())
                        else:
                            correctness = input(
                                'Would putting an %s in row %i and col'
                                ' %i be correct? (enter for yes or'
                                ' "no" for no)? ' % (player, row, col))
                            correctness = correctness == ""

                        if correctness:
                            assert current_sai == game.mark(row, col, player)
                            correctness = 1

                        else:
                            correctness = -1

                    agent.train(original_state, current_sai,
                                reward=correctness, next_state=game.machine_rep())

                except ValueError:
                    print("############################")
                    print("# Invalid move, try again. #")
                    print("############################")
                    #agent.train(original_state, 'Cell-%i-%i' % (row, col),
                               #'mark',
                                #{'value': player}, -1, 'mark', [])
            else:
                row, col, player = game.get_random_move()
                # row, col, player = best_move(game.list_rep())
                game.mark(row, col, player)

        winner = game.winner()
        if winner == "DRAW":
            print("draw")
            pass
        elif winner == agent_role:
            print("learning agent won")
            pass
        else:
            print("teaching agent won")
            pass
        if use_oracle:
            if game_count < max_games:
                new_game = True
        else:
            new = input("Play again? Press enter to continue or type 'no' to"
                        " stop.")
            new_game = new == ""

    return agent


if __name__ == "__main__":
    default = 'soartech'
    # default = 'Memo'
    # play_game_manually()

    parser = argparse.ArgumentParser(description='An interactive training demo'
                                                 'for apprentice agents.')
    parser.add_argument('-agent',
                        choices=['Modular', 'RLAgent', 'Memo', 'soartech'],
                        default=default, help='The agent type to use')

    args = parser.parse_args()

    if args.agent == 'soartech':
        agent = SoarTechAgent(wm=ExpertaWorkingMemory(ke=TTTEngineBasic()))
    elif args.agent == 'Memo':
        agent = Memo()
    elif args.agent == 'RLAgent':
        agent_class = RLAgent

    # agent = agent_class()
    train_agent(agent)
