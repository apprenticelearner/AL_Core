from apprentice.agents import SoarTechAgent
from apprentice.working_memory import ExpertaWorkingMemory
from apprentice.learners.when_learners import q_learner
from ttt_simple import ttt_engine, ttt_oracle
from pprint import pprint


if __name__ == "__main__":
    # with experta knowledge engine
    wm1 = ExpertaWorkingMemory(ke=ttt_engine())
    a1 = SoarTechAgent(
        wm=wm1, when=q_learner.QLearner(func=q_learner.Cobweb, q_init=0.6)
        # wm=wm1, when=q_learner.QLearner(func=q_learner.LinearFunc,
        #                                 q_init=0.6)
    )

    new_game = True
    while new_game:
        game = ttt_oracle()
        winner = False
        last_state = None
        last_sai = None

        while not winner:
            print()
            print("Current Player: " + game.current_player)
            print(game)
            state = game.as_dict()
            # pprint(state)

            if game.current_player == "X":
                if last_state is not None and last_sai is not None:
                    a1.train(last_state, state, last_sai, 0.0, "", [""])

                last_state = state
                sai = a1.request(state)
                last_sai = sai

                getattr(game, sai.action)(**sai.input)
                print("AI's move", sai)

            else:
                while True:
                    try:
                        loc = input("Enter move as row and column integers "
                                    "(e.g., 1,2):")
                        loc = loc.split(',')

                        row = int(loc[0])
                        col = int(loc[1])
                        player = game.current_player
                        game.move(row, col, player)
                        break
                    except Exception:
                        print("error with input, try again.")

            winner = game.check_winner()

        if winner == "X":
            a1.train(last_state, None, last_sai, 1.0, "", [""])
        elif winner == "O":
            a1.train(last_state, None, last_sai, -1.0, "", [""])
        else:
            a1.train(last_state, None, last_sai, 0, "", [""])

        print("WINNER = ", winner)
        print(game)
        print()

        new = input("Play again? Press enter to continue or type 'no' to"
                    " stop.")
        new_game = new == ""
