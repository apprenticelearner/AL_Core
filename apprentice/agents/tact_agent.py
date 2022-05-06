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
        self.sais_to_remove = []
        self.ext_facts: Dict[str, Fact] = {}
        self.matching_sais: List[Sai] = []
        self.max_depth = 1
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

    def add_skill(self, fact):
        self.skills.append(fact)
        self.add_fact(fact)

    def add_concept(self, fact):
        self.concepts.append(fact)
        self.add_fact(fact)

    def remove_skill(self, fact):
        self.skills.remove(fact)
        self.remove_fact(fact)

    def remove_concept(self, fact):
        self.concepts.remove(fact)
        self.remove_fact(fact)

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

    def get_new_skill_match(self):
        for pnode in self.pnodes:
            if pnode.new:
                t = pnode.pop_new_token()
                return Match(pnode, t)
        return None

    def get_new_concept_match(self):
        for pnode in self.pnodes:
            if pnode.new:
                t = pnode.pop_new_token()
                return Match(pnode, t)
        return None

    def explain(self, sai):
        print('trying to explain', sai)
        # for m in list(self.wm.matches):

        for s in self.get_sais_to_remove():
            self.matching_sais.remove(s)


        # for f in self.facts.values():
        #     d = 0
        #     if hasattr(f, 'depth'):
        #         d = f.depth
        #     print(d, f)

        for s in self.matching_sais:
            if s == sai:
                # print("EXPLAINATION DEPTH", s.depth)
                # print(s)
                self.render_trace(s)
                assert False
                return s

        # self.wm.render_graph()

        # Get all matches
        skill_matches = []
        concept_matches = []
        while True:
            m = self.get_new_match()
            if not m:
                break
            else:
                if m.mtype == 'skill':
                    skill_matches.append(m)
                else:
                    concept_matches.append(m)

        #while True:
        for m in concept_matches:
            # skip matches beyond depth
            depth = (sum([f.depth if hasattr(f, 'depth') else 0 for f in
                     self.get_dependent_facts(m)]) + 1)

            if (depth > self.max_depth and 'SAI' not in
                    m.pnode.production.__name__):
                continue

            try:
                output = m.fire()
            except Exception:
                # if rule fails, then it is an invalid match
                continue

            # if depth > self.max_depth and not isinstance(output, Sai):
            #     continue

            # print([(f.depth, f['id']) if hasattr(f, 'depth') else 0 for f in
            #        self.get_dependent_facts(m)])
            # print(depth)

            self.track_match_deletion(m)

            if isinstance(output, Sai):
                # print('adding sai', output)
                output.match = m
                output.depth = depth
                self.matching_sais.append(output)
                self.match_dependencies[m.token] = [output]
                if output == sai:
                    self.render_trace(output)
                    assert False
                    return sai
            elif output:
                for f in output:
                    # print('adding', f)
                    f.match = m
                    f.depth = depth
                    self.add_fact(f)
                self.match_dependencies[m.token] = output

        for m in skill_matches:
            # skip matches beyond depth
            depth = (sum([f.depth if hasattr(f, 'depth') else 0 for f in
                     self.get_dependent_facts(m)]) + 1)

            if (depth > self.max_depth and 'SAI' not in
                    m.pnode.production.__name__):
                continue

            try:
                output = m.fire()
            except Exception:
                # if rule fails, then it is an invalid match
                continue

            # if depth > self.max_depth and not isinstance(output, Sai):
            #     continue

            # print([(f.depth, f['id']) if hasattr(f, 'depth') else 0 for f in
            #        self.get_dependent_facts(m)])
            # print(depth)

            self.track_match_deletion(m)

            if isinstance(output, Sai):
                # print('adding sai', output)
                output.match = m
                output.depth = depth
                self.matching_sais.append(output)
                self.match_dependencies[m.token] = [output]
                if output == sai:
                    self.render_trace(output)
                    assert False
                    return sai
            elif output:
                for f in output:
                    # print('adding', f)
                    f.match = m
                    f.depth = depth
                    self.add_fact(f)
                self.match_dependencies[m.token] = output

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
    # This gets added to a token's children list in the rete, this makes it
    # possible to track when tokens and corresponding matches are deleted.

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
            skills: List[Production] = [],
            **kwargs
    ):
        # Just track the state as a set of Facts?
        # initialize to None, so gets replaced on first state.
        super().__init__()
        self.last_match = None
        self.last_sai: Optional[Sai] = None
        self.wm = WorkingMemory()

        log.debug(skills)

        for p in skills:
            self.wm.add_production(p)

    def request_diff(self, state_diff: Dict):
        """
        Queries the agent to get an action to execute in the world given the
        diff from the last state.

        :param state_diff: a state diff output from JSON Diff.
        """
        # Just loads in the differences from the state diff

        self.wm.update_with_diff(state_diff)
        return {}

        # print("FACTS")
        # for f in self.wm.facts:
        #     print(f, self.wm.facts[f])
        # print()

        # from pprint import pprint
        # pprint(self.wm.recently_removed_matches)

        # pprint(self.wm.match_dependencies)

        # print("# prod: {}".format(len(self.wm.productions)))
        # print("# facts: {}".format(len(self.wm.facts)))
        # print("# wmes: {}".format(len(self.wm.working_memory)))
        # print("# nodes: {}".format(self.wm.num_nodes()))
        # self.wm.render_graph()
        # print(self.wm)

        # output = None
        # candidate_activations = list(self.wm.matches)

        # print(candidate_activations)

        # while True:
        #     if len(candidate_activations) == 0:
        #         return {}

        #     best_match = random.choice(candidate_activations)
        #     candidate_activations.remove(best_match)

        #     # TODO update
        #     # state = self.wm.state

        #     output = best_match.fire()

        #     if isinstance(output, Sai):
        #         break

        #     candidate_activations = list(self.wm.matches)

        #     # TODO update
        #     # next_state = self.wm.state

        # self.last_match = best_match
        # self.last_sai = output

        # return {'selection': output.selection,
        #         'action': output.action,
        #         'inputs': output.inputs}

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

        explaination = self.wm.explain(sai)
        print('my exp is', explaination)

        if explaination is not None and reward > 0:
            print('learn new rule')

        elif explaination is None and reward > 0:
            print('memorize output')
            current_facts = [Fact(**{k: f[k] for k in f if k in {'id', 'value',
                             'contentEditable'}}) for f in
                             self.wm.facts.values()]

            @Production(AND(*current_facts))
            def new_rule():
                new_sai = Sai(selection=sai.selection,
                              action=sai.action,
                              inputs=sai.inputs)
                return new_sai

            # print(new_rule)

            # self.wm.add_production(new_rule)

        # need to apply both updates
        # print(next_state_diff)
        self.wm.update_with_diff(next_state_diff)

    def check(self, state: Dict, sai: Sai, **kwargs) -> float:
        """
        Checks the correctness (reward) of an SAI action in a given state.
        """
        raise NotImplementedError("Check not implemented for TACT agent.")


if __name__ == "__main__":
    pass
