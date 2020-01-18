from typing import List

from sklearn.feature_extraction import DictVectorizer


class FractionsStateHasher(object):

    def __init__(self):
        state_extremes = [{"('contentEditable', 'JCommTable.R0C0')": 'False',
                           "('contentEditable', 'JCommTable.R1C0')": 'False',
                           "('contentEditable', 'JCommTable2.R0C0')": 'False',
                           "('contentEditable', 'JCommTable3.R0C0')": 'False',
                           "('contentEditable', 'JCommTable3.R1C0')": 'False',
                           "('contentEditable', 'JCommTable4.R0C0')": 'False',
                           "('contentEditable', 'JCommTable4.R1C0')": 'False',
                           "('contentEditable', 'JCommTable5.R0C0')": 'False',
                           "('contentEditable', 'JCommTable5.R1C0')": 'False',
                           "('contentEditable', 'JCommTable6.R0C0')": 'False',
                           "('contentEditable', 'JCommTable6.R1C0')": 'False',
                           "('contentEditable', 'JCommTable7.R0C0')": 'False',
                           "('contentEditable', 'JCommTable8.R0C0')": 'False',
                           "('value', 'JCommTable2.R0C0')": '*',
                           "('value', 'JCommTable7.R0C0')": '*',
                           "('r_val', 'equal(JCommTable.R0C0, JCommTable.R0C0)')": 'True',
                           },
                          {"('contentEditable', 'JCommTable.R0C0')": 'True',
                           "('contentEditable', 'JCommTable.R1C0')": 'True',
                           "('contentEditable', 'JCommTable2.R0C0')": 'True',
                           "('contentEditable', 'JCommTable3.R0C0')": 'True',
                           "('contentEditable', 'JCommTable3.R1C0')": 'True',
                           "('contentEditable', 'JCommTable4.R0C0')": 'True',
                           "('contentEditable', 'JCommTable4.R1C0')": 'True',
                           "('contentEditable', 'JCommTable5.R0C0')": 'True',
                           "('contentEditable', 'JCommTable5.R1C0')": 'True',
                           "('contentEditable', 'JCommTable6.R0C0')": 'True',
                           "('contentEditable', 'JCommTable6.R1C0')": 'True',
                           "('contentEditable', 'JCommTable7.R0C0')": 'True',
                           "('contentEditable', 'JCommTable8.R0C0')": 'True',
                           "('value', 'JCommTable2.R0C0')": '+',
                           "('value', 'JCommTable7.R0C0')": '+',
                           }]

        fields = ["JCommTable.R0C0", "JCommTable.R1C0", "JCommTable2.R0C0",
                  "JCommTable3.R0C0", "JCommTable3.R1C0", "JCommTable4.R0C0",
                  "JCommTable4.R1C0", "JCommTable5.R0C0", "JCommTable5.R1C0",
                  "JCommTable6.R0C0", "JCommTable6.R1C0", "JCommTable7.R0C0",
                  "JCommTable8.R0C0"]

        for field1 in fields:
            for field2 in fields:
                if field1 > field2:
                    continue
                state_extremes[0][
                    "('r_val', 'equal(" + field1 + ", " + field2 + ")')"] = 'False'
                state_extremes[0][
                    "('contentEditable', 'add(" + field1 + ", " + field2 + ")')"] = 'False'
                state_extremes[0][
                    "('contentEditable', 'multiply(" + field1 + ", " + field2 + ")')"] = 'False'

                state_extremes[1][
                    "('r_val', 'equal(" + field1 + ", " + field2 + ")')"] = 'True'

        # from pprint import pprint
        # pprint(state_extremes)
        # raise Exception("BEEP")

        self.dv = DictVectorizer()
        self.dv.fit(state_extremes)

    def transform(self, states: List[dict]):
        # t = self.dv.transform(states)
        # from pprint import pprint
        # print("BEFORE")
        # pprint(states)
        # print('AFTER')
        # pprint(self.dv.inverse_transform(t))
        # print()

        return self.dv.transform(states)


class FractionsActionHasher(object):

    def __init__(self):
        action_extremes = [
            # {
            #  'fact-0: id': 'JCommTable6.R1C0',
            #  'fact-1: id': 'JCommTable.R1C0',
            #  'rulename': 'update_answer_field'
            # }
        ]

        operators = ['multiply', 'add']
        for o in operators:
            action_extremes.append({'fact-0: operator': o,
                                    'fact-1: operator': o})

        fields = ["JCommTable.R0C0", "JCommTable.R1C0", "JCommTable2.R0C0",
                  "JCommTable3.R0C0", "JCommTable3.R1C0", "JCommTable4.R0C0",
                  "JCommTable4.R1C0", "JCommTable5.R0C0", "JCommTable5.R1C0",
                  "JCommTable6.R0C0", "JCommTable6.R1C0", "JCommTable7.R0C0",
                  "JCommTable8.R0C0"]

        for f in fields:
            action_extremes.append({'fact-0: id': f,
                                    'fact-1: id': f,
                                    'fact-0: ele1': f,
                                    'fact-1: ele2': f,
                                    })

        prods = []
        for f1 in fields:
            for f2 in fields:
                if f1 > f2:
                    continue
                prods.append('add(' + f1 + ', ' + f2 + ')')
                prods.append('multiply(' + f1 + ', ' + f2 + ')')

        for p in prods:
            action_extremes.append({'fact-0: id': p})

        rule_names = ['click_done', 'check', 'equal' 'update_convert_field',
                      'update_answer_field', 'add', 'multiply']
        for name in rule_names:
            action_extremes.append({'rulename': name})

        self.dv = DictVectorizer()
        self.dv.fit(action_extremes)

    def transform(self, actions: List[dict]):
        # t = self.dv.transform(actions)
        # from pprint import pprint
        # print("BEFORE")
        # pprint(actions)
        # print('AFTER')
        # pprint(self.dv.inverse_transform(t))
        # print(t.shape)
        # print()

        return self.dv.transform(actions)
