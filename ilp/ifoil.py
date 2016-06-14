from numbers import Number
import re

import pexpect

from ilp.base_ilp import BaseILP
from ilp.foil import Foil

class iFoil(BaseILP):

    def __init__(self, closed_world=True, max_tuples=100000):
        self.foil = Foil(closed_world, max_tuples)
        self.rules = {}

    def fit(self, T, X, y):
        self.rules = {}
        self.target_types = [self.foil.example_type]
        for i in range(len(T[0])):
            sT = [(t[i],) for t in T]
            self.foil.name = "foa%i" % i
            self.foil.fit(sT, X, y)
            self.target_types.append(self.foil.target_types[1])

            rules = set()
            for rule in self.foil.rules:
                rule = re.sub(r'(?P<left>[,(])(?P<var>[B-Z])(?P<right>[,)])',
                              '\g<left>\g<var>%i\g<right>' % i, rule)
                rules.add(rule)

            self.rules[self.foil.name] = rules
            #print(self.rules)

    def get_matches(self, X, constraints=None):

        if constraints is None:
            constraints = []

        for foa in self.rules:
            if len(self.rules[foa]) == 0:
                return 

        X = {self.foil.clean(a): self.foil.clean(X[a]) for a in X}

        val_types = {}

        # get the type info
        for rel in X:
            if isinstance(rel, str):
                arg = rel + "Val"
                if not isinstance(X[rel], bool):
                    if arg not in val_types:
                        val_types[arg] = set()
                    if not isinstance(X[rel], Number):
                        val_types[arg].add(X[rel])
            elif isinstance(rel, tuple):
                for ele_id, ele in enumerate(rel[1:]):
                    arg = rel[0] + "Arg%i" % ele_id
                    if arg not in val_types:
                        val_types[arg] = set()
                    if ele in X:
                        val_types[arg].add(X[ele])
                    else:
                        val_types[arg].add(ele)
                if not isinstance(X[rel], bool):
                    arg = rel[0] + "Val"
                    if arg not in val_types:
                        val_types[arg] = set()
                    if not isinstance(X[rel], Number):
                        val_types[arg].add(X[rel])
            else:
                raise Exception("attribute not string or tuple.")

        # add type info to logic program
        #p.expect("\?- ")
        #p.sendline("assert(type" + self.example_type + "(" + self.example_name
        #           + ")).")
        data = self.foil.type_mapping[self.foil.example_type] + "(" + self.foil.example_name + ").\n"

        for tt in val_types:
            if val_types[tt] != "continuous":
                for val in val_types[tt]:
                    if tt in self.foil.type_mapping:
                        data += self.foil.type_mapping[tt] + "(" + val + ").\n"
                    else:
                        data += tt + "(" + val + ").\n"

                    #p.expect("\?- ")
                    #p.sendline("assert(type" + tt + "(" + val + ")).")

        for tt in self.foil.val_types:
            if "targetArg" in tt:
                if self.foil.val_types[tt] != "continuous":
                    for val in self.foil.val_types[tt]:
                        data += self.foil.type_mapping[tt] + "(" + val + ").\n"
                        #p.expect("\?- ")
                        #p.sendline("assert(type" + tt + "(" + val + ")).")

        # add aux relations to logic program
        rel_dict = dict()
        for rel in X:
            if isinstance(rel, str):
                rel_str = rel + "(" + self.foil.example_name
                if not isinstance(X[rel], bool):
                    rel_str += ", " + str(X[rel]) 
                rel_str += ").\n"

                if rel not in rel_dict:
                    rel_dict[rel] = set()
                rel_dict[rel].add(rel_str)

            elif isinstance(rel, tuple):
                rel_str = rel[0] + "(" + self.foil.example_name + ", " 
                #data = "assert(" + rel[0] + "(" + self.example_name + ", " 
                vals = [X[attr] if attr in X else attr for attr in rel[1:]]
                rel_str += ", ".join(vals)
                if not isinstance(X[rel], bool):
                    rel_str += ", " + str(X[rel]) 
                rel_str += ").\n"

                if rel[0] not in rel_dict:
                    rel_dict[rel[0]] = set()
                rel_dict[rel[0]].add(rel_str)

            else:
                raise Exception("attribute not string or tuple.")

        for rel in rel_dict:
            for rel_str in rel_dict[rel]:
                data += rel_str

        for foa in self.rules:
            for rule in self.rules[foa]:
                rule = re.sub(" (?P<first>\w+)<>(?P<second>\w+)", 
                              " dif(\g<first>, \g<second>)", rule)
                rule = re.sub(" not (?P<term>[\w,()]+),", 
                              " not(\g<term>),", rule)
                rule = rule.replace("<=", "=<")
                data += rule + ".\n"

        #print(data)

        args = [chr(i + ord("B")) for i in
                         range(len(self.target_types) - 1)]
        #all_args = [self.example_name] + args

        bind_args = ['A'] + args
        data += "bind_relation(" + ",".join(bind_args) + ") :- "
        
        bind_body = []

        for i in range(len(self.rules)):
            foa = "foa%i" % i
            bind_body += [foa + "(" + ",".join(['A'] + [args[i]]) + ")"]

        bind_body += [self.foil.type_mapping[tt] + "(" + bind_args[i] + ")" 
                     for i,tt in enumerate(self.target_types)]

        # user provided constraints
        bind_body += constraints
        data += ", ".join(bind_body)
        data += ".\n" 

        #print(data)

        outfile = open("foil_binding_lp.pl", 'w')
        outfile.write(data)
        outfile.close()

        #p = PopenSpawn('swipl -q -s foil_binding_lp.pl', encoding="utf-8",
        #               maxread=100000)
        p = pexpect.spawn('swipl -q -s foil_binding_lp.pl', timeout=10,
                          echo=False, encoding="utf-8", maxread=1000000,
                          searchwindowsize=None)


        p.expect("\?- ")

        p.sendline("bind_relation(" + ", ".join(bind_args) + ").")
        
        patterns = []
        for arg in bind_args:
            patterns.append(arg + r" = " + r"(?P<" + arg + r">\w+)")
        arg_pattern = r",[\r\n ]+".join(patterns)

        mid_pattern = arg_pattern + r" "
        end_pattern = arg_pattern + r"."

        previous_matches = set()

        resp = -1
        while resp != 0:
            if len(bind_args) > 0:
                try:
                    resp = p.expect(["false.", "ERROR", "true.", mid_pattern,
                                     end_pattern])
                except Exception as e:
                    print(e)
                    print(p.buffer)
                    print(p.before)
            else:
                resp = p.expect(["false.", "ERROR", "true."])

            if resp == 1:
                print(p)
                print(p.buffer)
                print(p.before)
                print(data)
                try:
                    raise Exception("Error in prolog program")
                except Exception:
                    return
            if resp == 2:
                yield tuple()
                return
            elif resp == 3 or resp == 4:
                mapping = {arg: p.match.group(arg) for arg in args}

                m = tuple([self.foil.unclean(self.foil.get_val(a, mapping)) 
                             for a in args])
                if m not in previous_matches:
                    yield m
                    previous_matches.add(m)

                if resp == 3:
                    p.send(";")
                else:
                    return
        

