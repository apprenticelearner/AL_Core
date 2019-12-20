from experta import Rule, Fact, W, KnowledgeEngine, MATCH
from random import randint

from apprentice.working_memory.representation import Sai


class RandomFracEngine(KnowledgeEngine):
    @Rule(
        Fact(id=MATCH.id, contentEditable=True, value=W())
    )
    def input_random(self, id):
        return Sai(selection=id, action='UpdateTextArea', input= {'value': str(randint(0,100))})

    @Rule(
        Fact(id='done')
    )
    def click_done(self):
        return Sai(selection='done', action='ButtonPressed', input={'value': -1})
