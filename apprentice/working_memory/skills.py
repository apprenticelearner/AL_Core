from random import randint

from apprentice.working_memory.adapters.experta_.factory import ExpertaSkillFactory
from apprentice.working_memory.representation import Sai

max_depth = 1

fields = ["JCommTable.R0C0", "JCommTable.R1C0", "JCommTable2.R0C0",
          "JCommTable3.R0C0", "JCommTable3.R1C0", "JCommTable4.R0C0",
          "JCommTable4.R1C0", "JCommTable5.R0C0", "JCommTable5.R1C0",
          "JCommTable6.R0C0", "JCommTable6.R1C0", "JCommTable7.R0C0",
          "JCommTable8.R0C0"]
answer_field = ['JCommTable6.R0C0', 'JCommTable6.R1C0']


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
        print('clicking done')
        return Sai(selection='done',
                   action='ButtonPressed',
                   input={'value': -1})
        # input={'value': '-1'})

    @Rule(
        Fact(id="JCommTable8.R0C0", contentEditable=True, value="")
    )
    def check(self):
        print('checking box')
        return Sai(selection="JCommTable8.R0C0",
                   action='UpdateTextArea',
                   input={'value': "x"})

    @Rule(
        Fact(id=MATCH.id1, contentEditable=False, value=MATCH.value1),
        TEST(lambda id1, value1: id1 in fields and value1 != ""),
        Fact(id=MATCH.id2, contentEditable=False, value=MATCH.value2),
        TEST(lambda id2, value2: id2 in fields and value2 != ""),
        TEST(lambda id1, id2: id1 < id2),
        NOT(Fact(relation='equal', ele1=MATCH.id1, ele2=MATCH.id2))
    )
    def equal(self, id1, value1, id2, value2):
        new_id = "equal(%s, %s)" % (id1, id2)
        equality = value1 == value2
        print('declaring equality', id1, id2, equality)
        self.declare(Fact(id=new_id,
                          relation='equal',
                          ele1=id1,
                          ele2=id2,
                          r_val=equality))

    @Rule(
        Fact(id='JCommTable8.R0C0', contentEditable=False, value='x'),
        Fact(id=W(), contentEditable=False, value=MATCH.value),
        TEST(lambda value: value != "" and is_numeric_str(value)),
        Fact(id=MATCH.field_id, contentEditable=True, value=W()),
        TEST(lambda field_id: field_id != 'JCommTable8.R0C0' and field_id not
             in answer_field),
    )
    def update_convert_field(self, field_id, value):
        print('updating convert field', field_id, value)
        return Sai(selection=field_id,
                   # action='UpdateTextField',
                   action='UpdateTextArea',
                   input={'value': value})

    @Rule(
        Fact(id=W(), contentEditable=False, value=MATCH.value),
        TEST(lambda value: value != "" and is_numeric_str(value)),
        Fact(id=MATCH.field_id, contentEditable=True, value=W()),
        TEST(lambda field_id: field_id != 'JCommTable8.R0C0'),
        TEST(lambda field_id: field_id in answer_field)
    )
    def update_answer_field(self, field_id, value):
        print('updating answer field', field_id, value)
        return Sai(selection=field_id,
                   # action='UpdateTextField',
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
        TEST(lambda value2: is_numeric_str(value2)),
        NOT(Fact(operator='add', ele1=MATCH.id1, ele2=MATCH.id2))
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

        print('adding', id1, id2)

        self.declare(Fact(id=new_id,
                          operator='add',
                          ele1=id1,
                          ele2=id2,
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
        TEST(lambda value2: is_numeric_str(value2)),
        NOT(Fact(operator='multiply', ele1=MATCH.id1, ele2=MATCH.id2))
    )
    def multiply(self, id1, value1, fact1, id2, value2, fact2):
        print('multiplying', id1, id2)
        new_id = 'multiply(%s, %s)' % (id1, id2)

        new_value = float(value1) * float(value2)
        if new_value.is_integer():
            new_value = int(new_value)
        new_value = str(new_value)

        depth1 = 0 if 'depth' not in fact1 else fact1['depth']
        depth2 = 0 if 'depth' not in fact2 else fact2['depth']
        new_depth = 1 + max(depth1, depth2)

        self.declare(Fact(id=new_id,
                          operator='multiply',
                          ele1=id1,
                          ele2=id2,
                          contentEditable=False,
                          value=new_value,
                          depth=new_depth))


ke = FractionsEngine()
skill_factory = ExpertaSkillFactory(ke)
click_done_skill = skill_factory.from_ex_rule(ke.click_done)
check_skill = skill_factory.from_ex_rule(ke.check)
equal_skill = skill_factory.from_ex_rule(ke.equal)
update_answer_field_skill = skill_factory.from_ex_rule(ke.update_answer_field)
update_convert_field_skill = skill_factory.from_ex_rule(
    ke.update_convert_field)
add_skill = skill_factory.from_ex_rule(ke.add)
multiply_skill = skill_factory.from_ex_rule(ke.multiply)

fraction_skill_set = {'click_done': click_done_skill, 'check': check_skill,
                      'update_answer': update_answer_field_skill,
                      'update_convert': update_convert_field_skill,
                      'equal': equal_skill,
                      'add': add_skill,
                      'multiply': multiply_skill}


class RandomFracEngine(KnowledgeEngine):
    @Rule(
        Fact(id=MATCH.id, contentEditable=True, value=W())
    )
    def input_random(self, id):
        return Sai(selection=id, action='UpdateTextArea',
                   input={'value': str(randint(0, 100))})

    @Rule(
        Fact(id='done')
    )
    def click_done(self):
        return Sai(selection='done', action='ButtonPressed',
                   input={'value': -1})


def fact_from_dict(f):
    if '__class__' in f:
        fact_class = f['__class__']
    else:
        fact_class = Fact
    f2 = {k: v for k, v in f.items() if k[:2] != "__"}
    return fact_class(f2)


if __name__ == "__main__":
    from apprentice.explain.explanation import Explanation
    import inspect

    engine = AdditionEngine()

    engine.reset()

    f1 = Fact(id='JCommTable.R0C0', value='1', contentEditable=False)
    f2 = Fact(id='JCommTable.R1C0', value='2', contentEditable=False)
    f3 = Fact(id='JCommTable.R1C1', contentEditable=True, value='')

    engine.declare(f1)
    engine.declare(f2)
    engine.declare(f3)
    engine.run(10)
    sais = engine.sais

    print("===")
    ex = Explanation(engine.sais[0])
    nr = ex.new_rule

    pass
