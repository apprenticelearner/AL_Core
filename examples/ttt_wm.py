from apprentice.agents import SoarTechAgent
from apprentice.working_memory import ExpertaWorkingMemory
from ttt_simple import ttt_engine, ttt_oracle

if __name__ == "__main__":
    # with experta knowledge engine
    o = ttt_oracle()
    wm1 = ExpertaWorkingMemory(ke=ttt_engine())
    a1 = SoarTechAgent(wm=wm1)

    while not o.check_winner():
        d = o.as_dict()
        sai = a1.request(d)
        getattr(o, sai.action)(**sai.input)
#        if done
#            reward = 1 if win else 0
#        a1.train(d, o.as_dict(), sai, reward, None, None)
        print(o)

