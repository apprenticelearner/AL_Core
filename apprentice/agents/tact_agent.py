import logging
import random
from pprint import pformat
from abc import ABCMeta
# from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
import inspect
import numpy as np
import scipy

import colorama
import jsondiff

from apprentice.agents.diff_base import DiffBaseAgent
from apprentice.working_memory.representation import Sai
from py_rete import Production
from py_rete import ReteNetwork
from py_rete import Fact
from py_rete import AND
from py_rete import V
from py_rete import Filter
from py_rete import Bind
from py_rete import Cond
from py_rete.common import Match
from py_rete.common import gen_variable

colorama.init(autoreset=True)

log = logging.getLogger(__name__)


class WorkingMemory(ReteNetwork):

    def __init__(self):
        super().__init__()
        self.recently_removed_matches = []
        self.match_dependencies = {}
        self.prod_to_facts = {}
        self.sais_to_remove = []
        self.ext_facts: Dict[str, Fact] = {}
        self.matching_sais: List[Sai] = []
        self.concepts = []
        self.skills = []

    def update_with_diff(self, diff):
        for k in diff:
            if k is jsondiff.symbols.replace:
                keys = [k2 for k2 in self.ext_facts]
                for k2 in keys:
                    self.remove_fact(self.ext_facts[k2])
                    del self.ext_facts[k2]
                for k2, v in diff[k].items():
                    new_fact = Fact(**v)
                    self.ext_facts[k2] = new_fact
                    self.add_fact(new_fact)
            elif k is jsondiff.symbols.delete:
                for k2 in diff[k]:
                    self.remove_fact(self.ext_facts[k2])
                    del self.ext_facts[k2]
            elif k in self.ext_facts:
                f = self.ext_facts[k]
                for attr in diff[k]:
                    f[attr] = diff[k][attr]
                self.update_fact(f)
            else:
                new_fact = Fact(**diff[k])
                self.ext_facts[k] = new_fact
                self.add_fact(new_fact)

    def get_new_skill_match(self) -> Optional[Match]:
        for skill in self.skills:
            for pnode in skill.p_nodes:
                if pnode.new:
                    # these use pop(0) so we get breadth first expansion
                    # i.e., (first in first out search)
                    t = pnode.new.pop(0)
                    return Match(pnode, t)
        return None

    def get_new_concept_match(self) -> Optional[Match]:
        for concept in self.concepts:
            for pnode in concept.p_nodes:
                if pnode.new:
                    t = pnode.new.pop(0)
                    return Match(pnode, t)
        return None

    def add_skill(self, skill):
        self.skills.append(skill)
        self.add_production(skill)

    def add_concept(self, concept):
        self.concepts.append(concept)
        self.add_production(concept)

    def remove_skill(self, skill):
        self.skills.remove(skill)
        self.remove_production(skill)
        # TODO remove any facts that depend on this skill

    def remove_concept(self, concept):
        self.concepts.remove(concept)
        self.remove_production(concept)
        # TODO remove any facts that depend on this concept

    def add_fact(self, fact):
        super().add_fact(fact)
        self.retract_dependent()

    def remove_fact(self, fact):
        super().remove_fact(fact)
        self.retract_dependent()

    def track_match_deletion(self, match):
        MatchDeletionTracker(self, match)

    def retract_dependent(self):
        while self.recently_removed_matches:
            m = self.recently_removed_matches.pop()
            # if m.token not in self.match_dependencies:
            #     continue

            for e in self.match_dependencies[m.token]:
                if isinstance(e, Fact):
                    # only remove facts that are still in wm
                    if e.id is not None:
                        super().remove_fact(e)

                elif isinstance(e, Sai):
                    self.sais_to_remove.append(e)
                else:
                    raise ValueError("Unknown type of element to remove")

            del self.match_dependencies[m.token]

    def get_sais_to_remove(self):
        while self.sais_to_remove:
            yield self.sais_to_remove.pop()

    def get_dependent_facts(self, match):
        ids = set([wme.identifier for wme in match.token.wmes])
        return [self.facts[i] for i in ids]

    def conceptual_inference(self):
        while True:
            m = self.get_new_concept_match()
            if not m:
                break

            try:
                output = m.fire()
            except Exception:
                # if rule fails, then it is an invalid match
                continue

            self.track_match_deletion(m)

            if output:
                print(output)

                for f in output:
                    # print('adding', f)
                    f.match = m
                    f.depth = 0
                    self.add_fact(f)
                self.match_dependencies[m.token] = output

    def skill_inference(self, max_depth=0):

        for s in self.get_sais_to_remove():
            self.matching_sais.remove(s)

        for s in self.matching_sais:
            if (s.depth > max_depth and 'SAI' not in
                    s.match.pnode.production.__name__):
                continue
            yield s

        while True:
            m = self.get_new_skill_match()
            if not m:
                break

            depth = (sum([f.depth if hasattr(f, 'depth') else 0 for f in
                     self.get_dependent_facts(m)]) + 1)

            if (depth > max_depth and 'SAI' not in
                    m.pnode.production.__name__):
                continue

            try:
                output = m.fire()
            except Exception:
                # if rule fails, then it is an invalid match
                continue

            self.track_match_deletion(m)

            if isinstance(output, Sai):
                # print('adding sai', output)
                output.match = m
                output.depth = depth
                self.matching_sais.append(output)
                self.match_dependencies[m.token] = [output]
                # Keep track of facts associated with skill/production
                self.prod_to_facts
                yield output

            elif output:
                for f in output:
                    # print('adding', f)
                    f.match = m
                    f.depth = depth
                    self.add_fact(f)
                # Facts produced from the firing of a rule depend on current match
                self.match_dependencies[m.token] = output

    def explain(self, sai, max_depth=0):
        for output in self.skill_inference(max_depth=max_depth):
            if output == sai:
                # self.render_trace(output)
                # assert False
                yield output

    def render_trace(self, ele):
        print("CONDITIONS OF COMPILE")
        print(self.compile_explaination(ele))
        print("END")
        assert False

        print(ele.match.token.wmes)
        print(ele.match.pnode.production.get_rete_conds())

        import networkx as nx
        from networkx.drawing.nx_agraph import graphviz_layout
        import matplotlib.pyplot as plt

        def get_dependent_facts(wm, match):
            ids = set([wme.identifier for wme in match.token.wmes])
            return [wm.facts[i] for i in ids]

        def get_label(ele):
            if isinstance(ele, Sai):
                return str(ele)
            if isinstance(ele, Fact):
                return "{}: {}".format(ele.id, ele)
            if isinstance(ele, Match):
                kwargs = {arg: ele.pnode.production._rete_net.facts[ele.token.binding[V(arg)]] if
                          ele.token.binding[V(arg)] in ele.pnode.production._rete_net.facts else
                          ele.token.binding[V(arg)]
                          for arg in ele.pnode.production._wrapped_args}
                func_name = ele.pnode.production.__wrapped__.__name__
                signature = inspect.signature(ele.pnode.production.__wrapped__)

                return "{}{}: {}".format(func_name, signature, kwargs)
                return "{}".format(ele.pnode.production.pattern)
                return "{}: {}".format(id(ele.token), ele.token)
                # return "{}: {}".format(id(ele.token),
                #                        ele.pnode.production)

        G = nx.DiGraph()

        ele_to_render = [ele]
        while ele_to_render:
            ele = ele_to_render.pop()

            if hasattr(ele, 'match'):
                G.add_edge(get_label(ele.match), get_label(ele))
                for f in get_dependent_facts(self, ele.match):
                    G.add_edge(get_label(f), get_label(ele.match))
                    ele_to_render.append(f)

        # put rest of trace here?

        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, with_labels=True, font_size=8)
        # nx.draw(G, with_labels=True, font_weight="bold")
        plt.show()

    def compile_explaination(self, ele):
        def get_dependent_facts(wm, match):
            ids = set([wme.identifier for wme in match.token.wmes])
            return [wm.facts[i] for i in ids]

        conditions = []
        code = []
        ele_to_regress = [ele]

        while ele_to_regress:
            ele = ele_to_regress.pop()

            if hasattr(ele, 'match'):
                code_nodes = 0

                # print(ele.match.token.binding)

                wmes = list(ele.match.token.wmes)
                new_bindings = {}
                for i, c in enumerate(
                        ele.match.pnode.production.get_rete_conds()[0]):
                    if isinstance(c, Filter):
                        code_nodes += 1
                        args = inspect.getfullargspec(c.func)[0]
                        failed = False
                        for a in args:
                            if V(a) not in new_bindings:
                                failed = True
                                break

                        if failed:
                            # print('failed code node')
                            continue

                        arg_str = ",".join([new_bindings[V(a)].name
                                            for a in args])

                        f = "lambda "
                        f += arg_str + ", compiled_f=c.func"
                        f += ": compiled_f(" + arg_str + ")"
                        # print(f)
                        f = eval(f, globals(), locals())
                        conditions.append(Filter(f))

                    elif isinstance(c, Cond):
                        if not hasattr(self.facts[
                                wmes[i-code_nodes].identifier], 'match'):
                            # conditions.append(wmes[i-code_nodes])
                            new_id = c.identifier
                            if isinstance(new_id, V):
                                if new_id not in new_bindings:
                                    new_bindings[new_id] = gen_variable()
                                new_id = new_bindings[new_id]

                            new_attr = c.attribute
                            if isinstance(new_attr, V):
                                if new_attr not in new_bindings:
                                    new_bindings[new_attr] = gen_variable()
                                new_attr = new_bindings[new_attr]

                            new_val = c.value
                            if isinstance(new_val, V):
                                if new_val not in new_bindings:
                                    new_bindings[new_val] = gen_variable()
                                new_val = new_bindings[new_val]

                            # print()
                            # print(c, 'vs', wmes[i-code_nodes])
                            # print(Cond(new_id, new_attr, new_val))
                            # print()
                            conditions.append(Cond(new_id, new_attr, new_val))
                    else:
                        raise ValueError("Compliation not supported for this"
                                         " kind of condition {}".format(c))

                for f in get_dependent_facts(self, ele.match):
                    ele_to_regress.append(f)

                # build function exec
                code.append(ele)

        print(new_bindings)
        for ele in code:
            print('ele', ele)

        @Production(conditions)
        def new_prod():
            for ele in code:
                facts = ele.match.pnode.production.func
                print(facts)

        # print()
        # print("NEW PRODUCTION")
        # print(new_prod)
        # print()

        return conditions


class MatchDeletionTracker:
    """
    This gets added to a token's children list in the rete, this makes it
    possible to track when tokens and corresponding matches are deleted.
    """

    def __init__(self, wm, match):
        self.wm = wm
        self.match = match
        self.match.token.children.append(self)

    def delete_token_and_descendents(self):
        self.wm.recently_removed_matches.append(self.match)
        self.match.token.children.remove(self)


class TACTAgent(DiffBaseAgent):
    """
    Agent that uses PyRete
    """

    def __init__(
            self,
            concepts: List[Production] = None,
            skills: List[Production] = None,
            **kwargs
    ):
        # Just track the state as a set of Facts?
        # initialize to None, so gets replaced on first state.
        super().__init__()
        self.last_match = None
        self.last_sai: Optional[Sai] = None
        self.wm = WorkingMemory()
        self.exp_to_skills = {}
        self.var_mappings = {}
        self.var_counter = 0

        log.debug(concepts)
        log.debug(skills)

        if concepts:
            for c in concepts:
                self.wm.add_concept(c)

        if skills:
            for s in skills:
                self.wm.add_skill(s)

    def request_diff(self, state_diff: Dict):
        """
        Queries the agent to get an action to execute in the world given the
        diff from the last state.

        :param state_diff: a state diff output from JSON Diff.
        """
        # Just loads in the differences from the state diff

        self.wm.update_with_diff(state_diff)

        self.wm.conceptual_inference()

        for output in self.wm.skill_inference():
            self.last_sai = output
            return {'selection': output.selection,
                    'action': output.action,
                    'inputs': output.inputs}

        return {}

    def compile_memo_rule(self, sai):
        current_facts = [Fact(**{k: f[k] for k in f})
                         for f in self.wm.facts.values()]

        @Production(AND(*current_facts))
        def memoSAI_rule():
            return Sai(selection=sai.selection,
                       action=sai.action,
                       inputs=sai.inputs)

        return memoSAI_rule

    def update_skill_with_negative(self, skill):
        """
        restrict conditions on the provided skill so that it no longer match
        the current working memory state.
        """
        print('updating skill with negative not implemented.')
        # ANTI UNIFICATION - LOOKUP CONCEPT LEARNING
        # want to add additional condition
        current_facts = set(tuple((k, f[k]) for k in f)
                            for f in self.wm.facts.values())
        # Those facts that caused the skill to match
        intersection = []

        for f in skill.pattern:
            e = tuple((k, f[k]) for k in f)
            if e in current_facts:
                intersection.append(f)


    def update_skill_with_positive(self, skill):
        """
        generalize conditions on the provided skill so that it matches
        the current working memory state.
        """

        wm_facts = list(self.wm.facts.values())
        skill_facts = [f for f in skill.pattern]
        # List of antiunified facts
        new_skill_facts = self.antiunify_sets(wm_facts, skill_facts)
        #exit(0)
        # Update skill by removing and re-adding
        #for i in self.wm.skills: print(i)
        #print('OUR SKILL')
        #print(skill)
        #self.wm.remove_skill(skill)


    def train_diff(self, state_diff, next_state_diff, sai, reward):
        """
        Need the diff for the current state as well as the state diff for
        computing the state that results from taking the action. This is
        needed for performing Q learning.

        Accepts a JSON object representing the state, a string representing the
        skill label, a list of strings representing the foas, a string
        representing the selection, a string representing the action, list of
        strings representing the inputs, and a boolean correctness.
        """
        # need to apply both updates to get from
        # prior_state->state->next_state
        self.wm.update_with_diff(state_diff)

        if reward < 0:
            for explaination in self.wm.explain(sai, max_depth=0):
                # restrict conditions on matching rules so that
                # they no longer match the current situation
                self.update_skill_with_negative(
                        explaination.match.pnode.production)
        else:
            explaination = None
            #print('printing SAI in train_diff')
            #print(sai)
            for explaination in self.wm.explain(sai, max_depth=1):

                if explaination.depth == 1:
                    # we have a rule (one-step exp) that matches and is
                    # correct, do nothing?
                    break

                if explaination in self.exp_to_skills:
                    # we have an explaination that corresponds to an existing
                    # rule (skill) however the rule does not match, generalize
                    # conditions to make it match
                    self.update_skill_with_positive(
                            self.exp_to_skills[explaination])
                    break

                else:
                    # we have an explaination that does not correspond to an
                    # existing rule, compile a new rule
                    print("compile new rule here.")
                    new_skill = self.compile_explaination(explaination)
                    self.wm.add_skill(new_skill)
                    self.exp_to_skills[explaination] = new_skill
                    break

            if explaination is None:
                #print('explaination is none')
                k = (sai.selection, sai.action, frozenset(sai.inputs.items()))
                if k in self.exp_to_skills:
                    # Check if memo rule already exists for SAI, if so, then
                    # generalize the rule to cover this instance.
                    self.update_skill_with_positive(
                            self.exp_to_skills[k])
                else:
                    # we are unable to find an explaination, create a memo rule
                    print('memorize state-action as memo rule')
                    new_skill = self.compile_memo_rule(sai)
                    self.wm.add_skill(new_skill)
                    self.exp_to_skills[k] = new_skill

        # need to apply both updates
        self.wm.update_with_diff(next_state_diff)

    def check(self, state: Dict, sai: Sai, **kwargs) -> float:
        """
        Checks the correctness (reward) of an SAI action in a given state.
        """
        raise NotImplementedError("Check not implemented for TACT agent.")

    def get_skolem(self):
        self.var_counter += 1
        return 'var-antiunify{}'.format(self.var_counter)

    def antiunify_facts(self, f1, f2):
        """
        Takes two facts and antiunifies them into a new fact. The new fact contains
        the intersection of attributes between f1 and f2, and values remain as
        constant values if both facts have the same attr:value pair, otherwise the
        value is replaced by a variable V.

        :param f1: Fact 1 to antiunify
        :param f2: Fact 2 to antiunify
        :param var_mappings: A dict that maps attribute values (x,y) to variable
                             definitions of type V
        :return: The antiunified fact or None, the specificity-reward of the antiunification,
                 the dict of variable mappings
        """
        # Case where either f1 or f2 are nil
        if f1 is None or f2 is None:
            return None, 0

        # Get new value pair to variable mappings if not in var_mappings already
        new_var_mappings = {(f1[attr], f2[attr]): V(self.get_skolem())
                            if (f1[attr], f2[attr]) not in self.var_mappings else self.var_mapping[(f1[attr], f2[attr])]
                            for attr in f1 if attr in f2
                            if f1[attr] != f2[attr]}

        self.var_mappings.update(new_var_mappings)

        # Create antiunified fact dict using constant values when values are the same
        # otherwise using the var_mapping dict
        new_fact_dict = {attr: f1[attr] if f1[attr] == f2[attr]
                                        else self.var_mappings[(f1[attr], f2[attr])]
                                        for attr in f1 if attr in f2}

        # For case where two facts have nothing in common, new_fact_dict is empty
        if not new_fact_dict:
            return None, 0

        # 1 for each attribute
        reward = len(new_fact_dict)
        reward += (sum([1 for attr in f1 if attr in f2 if f1[attr] != f2[attr]])
                      - len(new_var_mappings))
        # Add 1 for each attribute with constant value
        for val in new_fact_dict.values():
            if not isinstance(val, V):
                reward += 1

        return Fact(**new_fact_dict), reward


    # NEED TO AVOID GENERALIZING ON RELATIONS
    def antiunify_sets(self, wm_facts, skill_facts):
        self.var_mappings = {}
        # Need the cost matrix to be square, so pad shorter list with None values
        if len(wm_facts) != len(skill_facts):
            wm_facts, skill_facts = self.pad_with_null(wm_facts, skill_facts)
        # Used to store new antiunified facts
        new_facts = {}

        # Initialize cost matrix
        cost_matrix = np.zeros((len(skill_facts), len(wm_facts)))
        # Iterate over skill facts (rows of cost matrix) and WM facts (cols of cost matrix)
        # and antiunify facts one to one, save returned cost and fact
        for i, skill_fact in enumerate(skill_facts):
            for j, wm_fact in enumerate(wm_facts):
                new_fact, cost = self.antiunify_facts(skill_fact, wm_fact)
                if new_fact:
                    new_facts[(i,j)] = new_fact
                cost_matrix[i][j] = cost

        #print(f'COST MATRIX: {cost_matrix}')
        #print(f'COST MATRIX SHAPE: {np.shape(cost_matrix)}')

        rows, cols = scipy.optimize.linear_sum_assignment(cost_matrix, maximize=True)
        #print(f'ROW INDEX ASSIGNMENTS: {rows}')
        #print(f'COLUMN INDEX ASSIGNMENTS: {cols}')
        #values = {(rows[i], cols[i]): cost_matrix[rows[i]][cols[i]] for i, _ in enumerate(rows)}
        #print(f'COST MATRIX ASSIGNMENTS: {values}')
        new_skill_facts = [new_facts.get((rows[i], cols[i])) for i, _ in enumerate(rows)
                           if new_facts.get((rows[i], cols[i]))]

        #print(f'NEW FACTS: {new_skill_facts}')
        #print(f'NUM NEW FACTS: {len(new_skill_facts)}')
        return new_skill_facts

    def pad_with_null(self, list1, list2):
        def add_null(li, num):
            for i in range(num):
                li.append(None)
            return li

        list1_len = len(list1)
        list2_len = len(list2)

        diff = list1_len - list2_len
        if diff < 0:
            list1 = add_null(list1, np.abs(diff))
        else:
            list2 = add_null(list2, np.abs(diff))
        return list1, list2


if __name__ == "__main__":
    pass
