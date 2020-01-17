from apprentice.agents import SoarTechAgent
from apprentice.working_memory import ExpertaWorkingMemory
from apprentice.learners.when_learners import q_learner
from ttt_simple import ttt_engine, ttt_oracle

if __name__ == "__main__":
    # with experta knowledge engine
    wm1 = ExpertaWorkingMemory(ke=ttt_engine())
    a1 = SoarTechAgent(
        wm=wm1,
        # when=q_learner.QLearner(func=q_learner.Cobweb, q_init=0.0),
        feature_set=[], function_set=[],
        # wm=wm1, when=q_learner.QLearner(func=q_learner.LinearFunc, q_init=0.0),
        negative_actions=True, action_penalty=0.0, epsilon=0.05
    )

    max_training_games = 100
    consecutive_wins = 0
    prev_win_board = None
    i = 0
    while consecutive_wins < 5 and i < max_training_games:
        o = ttt_oracle()
        winner = False
        print("#############################")
        print("Training game {}".format(i))
        print("#############################")
        print()

        i += 1
        while not winner:
            state = o.as_dict()
            sai = a1.request(state)

            getattr(o, sai.action)(**sai.input)
            next_state = o.as_dict()
            winner = o.check_winner()
            reward = 0

            if winner is not False:
                next_state = None

            if winner == "X":
                reward = -1
                consecutive_wins = 0
            if winner == "O":
                reward = 1
                if prev_win_board and o.as_dict() == prev_win_board:
                    consecutive_wins += 1
                else:
                    prev_win_board = o.as_dict()
                    consecutive_wins = 0

            a1.train(state, next_state, sai, reward, "", [""])
        print("#############################")
        print("Final reward = %0.1f" % reward)
        print("#############################")
        print()

    test_games = 20
    a1.epsilon = 0
    wins = 0
    for i in range(test_games):
        o = ttt_oracle()
        print("test game {}".format(i))
        while not o.check_winner():
            d = o.as_dict()
            sai = a1.request(d)
            getattr(o, sai.action)(**sai.input)
        print(o)
        if o.check_winner() == "O":
            wins += 1
        if o.check_winner() == "X":
            wins -= 1

    print("#############################")
    print("test win rate: {}".format(wins / test_games))
    print("#############################")
    print()
