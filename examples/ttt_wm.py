from experta import *
from experta.conditionalelement import __all__, OperableCE

from ttt_simple import ttt_engine, ttt_oracle
from apprentice.working_memory import ExpertaWorkingMemory
from apprentice.agents import SoarTechAgent

if __name__ == "__main__":
    t = ttt_engine()
    o = ttt_oracle()
    wm = ExpertaWorkingMemory(ke=t)
    a = SoarTechAgent(wm=wm)



