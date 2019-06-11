from pprint import pprint

from agents.utils import tup_sai


class IncrementalMany(object):

    def __init__(self, planner):
        self.planner = planner
        self.explanations = {}
        self.examples = []

    def ifit(self, example):
        found = False
        for exp in self.explanations:
            if self.explains(exp, example):
                self.explanations[exp].append(example)
                found = True
    
        if not found and example['correct'] is True:
            sai = tup_sai(example['selection'], example['action'],
                          example['inputs'])
            exp = tuple(self.planner.explain_sai(example['limited_state'],
                                                 sai))[0]
            self.explanations[exp] = [example]
            for e in self.examples:
                if self.explains(exp, e):
                    self.explanations[exp].append(e)

        self.examples.append(example)
        self.remove_subsumed()
        return self.explanations

    def remove_subsumed(self):
        unnecessary = set()

        explanations = list(self.explanations)
        for i, exp1 in enumerate(explanations):
            for exp2 in explanations[i+1:]:
                if self.subsumes(exp1, exp2):
                    unnecessary.add(exp2)
                elif self.subsumes(exp2, exp1):
                    unnecessary.add(exp1)

        for exp in unnecessary:
            del self.explanations[exp]

    def subsumes(self, exp1, exp2):
        """
        Checks if one explanation subsumes another. However, this only applies
        to the positive examples. The negative examples are kept around in
        self.examples, but we don't need explainations that only cover negative
        examples.  
        """
        for e in self.explanations[exp2]:
            if e['correct'] is False:
                continue
            if e not in self.explanations[exp1]:
                return False
        return True

    def fit(self, examples):
        self.explanations = {}
        for e in examples:
            self.ifit(e)
        return self.explanations

    def explains(self, explanation, example):
        """
        Checks if an explanation successfully explains an example
        """
        try:
            sai = tup_sai(example['selection'], example['action'],
                          [example['inputs'][a] for a in example['inputs']])
            grounded_plan = tuple([self.planner.execute_plan(ele,
                                    example['limited_state'])
                                    for ele in explanation])
            print()
            print(sai, 'VS', grounded_plan)
            print()
            return self.planner.is_sais_equal(grounded_plan, sai)
        except Exception as e:
            print(e)
            print('plan could not execute')
            pprint(explanation)
            pprint(example['limited_state'])
            return False


class SimStudentHow(IncrementalMany):

    def ifit(self, example):
        found = False

        for exp in list(self.explanations):
            if self.explains(exp, example):
                self.explanations[exp].append(example)
                found = True
            elif not found and example['correct'] is True:
                seed = self.explanations[exp][0]
                sai = tup_sai(seed['selection'], seed['action'],
                              seed['inputs'])

                print("LIMITED STATE FOR HOW")
                pprint(seed['limited_state'])

                for new_exp in self.planner.explain_sai_iter(seed['limited_state'],
                                                        sai):
                    if not self.explains(new_exp, example):
                        continue
                    covers = True
                    for e in self.explanations[exp][1:]:
                        if e['correct'] and not self.explains(new_exp, e):
                            covers = False
                            break
                    if covers:
                        self.explanations[new_exp] = [example]
                        for e in self.examples:
                            if self.explains(new_exp, e):
                                self.explanations[new_exp].append(e)
                        found = True
                        break
        if not found and example['correct']:
            sai = tup_sai(example['selection'], example['action'],
                          example['inputs'])
            exp = tuple(self.planner.explain_sai(example['limited_state'],
                                                 sai))[0]
            self.explanations[exp] = [example]
            for e in self.examples:
                if self.explains(exp, e):
                    self.explanations[exp].append(e)

        self.examples.append(example)
        self.remove_subsumed()
        return self.explanations

def get_how_learner(name):
    return HOW_LEARNERS[name.lower().replace(' ', '').replace('_', '')]

HOW_LEARNERS = {
    'incremental':IncrementalMany,
    'simstudent':SimStudentHow
}
