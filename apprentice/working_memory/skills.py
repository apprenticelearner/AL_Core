from experta import Rule, Fact, W, KnowledgeEngine, MATCH, TEST, AS
from random import randint

from apprentice.working_memory.representation import Sai

max_depth = 1


def is_numeric_str(x):
    try:
        x = float(x)
        return True
    except Exception:
        return False


class FractionsEngine(KnowledgeEngine):

    @Rule(
        Fact(id='done')
    )
    def click_done(self):
        return Sai(selection='done',
                   action='ButtonPressed',
                   input={'value': -1})

    @Rule(
        Fact(id=W(), contentEditable=False, value=MATCH.value),
        TEST(lambda value_from: value_from != ""),
        Fact(id=MATCH.field_id, contentEditable=True, value=W()))
    def update_field(self, field_id, value):
        return Sai(selection=field_id,
                   action='UpdateTextArea',
                   input={'value': value})

    @Rule(
        AS.fact1 << Fact(id=MATCH.id1, contentEditable=False,
                         value=MATCH.value1),
        TEST(lambda fact1: 'depth' not in fact1 or fact1['depth'] < max_depth),
        TEST(lambda value1: is_numeric_str(value1)),
        AS.fact2 << Fact(id=MATCH.id2, contentEditable=False,
                         value=MATCH.value2),
        TEST(lambda id1, id2: id1 <= id2),
        TEST(lambda fact2: 'depth' not in fact2 or fact2['depth'] < max_depth),
        TEST(lambda value2: is_numeric_str(value2))
    )
    def add(self, id1, value1, fact1, id2, value2, fact2):
        new_id = 'add(%s, %s)' % (id1, id2)

        new_value = float(value1) + float(value2)
        if new_value.is_integer():
            new_value = int(new_value)
        new_value = str(new_value)

        depth1 = 0 if 'depth' not in fact1 else fact1['depth']
        depth2 = 0 if 'depth' not in fact2 else fact2['depth']
        new_depth = 1 + max(depth1, depth2)

        self.declare(Fact(id=new_id,
                          contentEditable=False,
                          value=new_value,
                          depth=new_depth))

    @Rule(
        AS.fact1 << Fact(id=MATCH.id1, contentEditable=False,
                         value=MATCH.value1),
        TEST(lambda fact1: 'depth' not in fact1 or fact1['depth'] < max_depth),
        TEST(lambda value1: is_numeric_str(value1)),
        AS.fact2 << Fact(id=MATCH.id2, contentEditable=False,
                         value=MATCH.value2),
        TEST(lambda id1, id2: id1 <= id2),
        TEST(lambda fact2: 'depth' not in fact2 or fact2['depth'] < max_depth),
        TEST(lambda value2: is_numeric_str(value2))
    )
    def subtract(self, id1, value1, fact1, id2, value2, fact2):
        new_id = 'subtract(%s, %s)' % (id1, id2)

        new_value = float(value1) - float(value2)
        if new_value.is_integer():
            new_value = int(new_value)
        new_value = str(new_value)

        depth1 = 0 if 'depth' not in fact1 else fact1['depth']
        depth2 = 0 if 'depth' not in fact2 else fact2['depth']
        new_depth = 1 + max(depth1, depth2)

        self.declare(Fact(id=new_id,
                          contentEditable=False,
                          value=new_value,
                          depth=new_depth))

    # @Rule(
    #     Fact(id=MATCH.id1, contentEditable=False, value=MATCH.value1),
    #     TEST(lambda value1: is_numeric_str(value1)),
    #     Fact(id=MATCH.id2, contentEditable=False, value=MATCH.value2),
    #     TEST(lambda value2: is_numeric_str(value2))
    # )
    # def multiply(self, id1, value1, id2, value2):
    #     new_value = float(value1) * float(value2)
    #     if new_value.is_integer():
    #         new_value = int(new_value)
    #     new_value = str(new_value)

    #     new_id = 'multiply(%s, %s)' % (id1, id2)

    #     self.declare(Fact(id=new_id,
    #                       contentEditable=False,
    #                       value=new_value))

    # @Rule(
    #     Fact(id=MATCH.id1, contentEditable=False, value=MATCH.value1),
    #     TEST(lambda value1: is_numeric_str(value1)),
    #     Fact(id=MATCH.id2, contentEditable=False, value=MATCH.value2),
    #     TEST(lambda value2: is_numeric_str(value2))
    # )
    # def divide(self, id1, value1, id2, value2):
    #     new_value = float(value1) / float(value2)
    #     if new_value.is_integer():
    #         new_value = int(new_value)
    #     new_value = str(new_value)

    #     new_id = 'divide(%s, %s)' % (id1, id2)

    #     self.declare(Fact(id=new_id,
    #                       contentEditable=False,
    #                       value=new_value))


class RandomFracEngine(KnowledgeEngine):
    @Rule(
        Fact(id=MATCH.id, contentEditable=True, value=W())
    )
    def input_random(self, id):
        return Sai(selection=id, action='UpdateTextArea', input={'value': str(randint(0, 100))})

    @Rule(
        Fact(id='done')
    )
    def click_done(self):
        return Sai(selection='done', action='ButtonPressed', input={'value': -1})
