from apprentice.agents import SoarTechAgent
from apprentice.working_memory import ExpertaWorkingMemory
from apprentice.learners.when_learners import q_learner
from ttt_simple import ttt_engine, ttt_oracle

if __name__ == "__main__":
    # with experta knowledge engine
    wm1 = ExpertaWorkingMemory(ke=ttt_engine())
    a1 = SoarTechAgent(wm=wm1)

#    o = ttt_oracle()
#    d = o.as_dict()
#    sai = a1.request(d)

#    pre_test_games = 25
#    wins = 0
#    for i in range(pre_test_games):
#        o = ttt_oracle()
#        print("Pretest game {}".format(i))
#        while not o.check_winner():
#            d = o.as_dict()
#            sai = a1.request(d)
#            getattr(o, sai.action)(**sai.input)
#        if o.check_winner() == "O":
#            wins += 1
#        if o.check_winner() == "X":
#            wins -= 1
#
#    print("Pretest win rate: {}".format(wins/pre_test_games))

    # train to win with O
    num_traning_games = 1050
    for i in range(num_traning_games):
        o = ttt_oracle()
        winner = False
        print("Training game {}".format(i))
        while not winner:
            state = o.as_dict()
            sai = a1.request(state)

            getattr(o, sai.action)(**sai.input)
            next_state = o.as_dict()
            winner = o.check_winner()
            reward = 0
            if winner == "X":
                reward = -1
            if winner == "O":
                reward = 1

            a1.train(state, next_state, sai, reward, "", [""])

    test_games = 20
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

    print("test win rate: {}".format(wins / test_games))




