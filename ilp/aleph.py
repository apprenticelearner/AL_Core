from pprint import pprint
from numbers import Number
from subprocess import Popen
from subprocess import PIPE
from subprocess import STDOUT
from itertools import product
import re

import pexpect

from ilp.base_ilp import BaseILP

class Aleph(BaseILP):

    def __init__(self):
        """
        The constructor. 
        """
        # Keywords used for constructing foil file (try not to conflict).
        self.name = "target_relation"
        self.example_type = "E"
        self.example_name = self.clean("curr_example")
        self.example_keyword = "e"

        # initialize all class variables
        self.initialize()

    def initialize(self):

        # the target relation types
        self.target_types = None

        # tuples for target relation
        self.pos = set()
        self.neg = set()

        # the possible attr value types
        # the val type "continuous" is reserved for numerical types
        self.val_types = {}

        # the aux relations and their type tuples
        # also the pos tuples for aux relations
        self.aux_relations = {}
        self.pos_aux_tuples = {}

        # String representation of learned rules
        self.rules = ""

    def fit(self, T, X, y):

        # initialize all variables for learning
        self.initialize()

        # clean the input
        T = [self.clean(t) for t in T]
        X = [{self.clean(a): self.clean(x[a]) for a in x} for x in X]

        self.val_types[self.example_type] = set()

        for i, correct in enumerate(y):
            example_id = "%s%i" % (self.example_keyword, i) 
            self.val_types[self.example_type].add(example_id)

            # Get target relation tuples
            if correct == 1:
                self.pos.add((example_id,) + T[i])
            else:
                self.neg.add((example_id,) + T[i])

            target_types = (self.example_type,)
            for ai, v in enumerate(T[i]):
                arg = "targetArg%i" % ai
                target_types += (arg,)

                if arg not in self.val_types:
                    self.val_types[arg] = set()
                self.val_types[arg].add(v)

            if self.target_types is None:
                self.target_types = target_types
            else:
                if self.target_types != target_types:
                    raise Exception("All tuples should have same types")

            # accumulate aux relation information
            for rel in X[i]:
                if isinstance(rel, str):
                    arg = rel + "Val"
                    if rel not in self.aux_relations:
                        if arg not in self.val_types:
                            self.val_types[arg] = set()

                        if isinstance(X[i][rel], bool):
                            self.aux_relations[rel] = (self.example_type,)
                        else:
                            self.aux_relations[rel] = (self.example_type, arg)

                        self.pos_aux_tuples[rel] = set()
                    if not isinstance(X[i][rel], bool):
                        self.val_types[arg].add(str(X[i][rel]))
                    self.pos_aux_tuples[rel].add((example_id, str(X[i][rel])))

                elif isinstance(rel, tuple):
                    tup_type = (self.example_type,)
                    for ele_id, ele in enumerate(rel[1:]):
                        arg = rel[0] + "Arg%i" % ele_id
                        if arg not in self.val_types:
                            self.val_types[arg] = set()
                        tup_type += (arg,)

                    if not isinstance(X[i][rel], bool):
                        if rel[0] + "Val" not in self.val_types:
                            self.val_types[rel[0] + "Val"] = set()
                        tup_type += (rel[0] + "Val",)

                    if rel[0] not in self.aux_relations:
                        self.aux_relations[rel[0]] = tup_type
                        self.pos_aux_tuples[rel[0]] = set()

                    tup = (example_id,)
                    for ai, attr in enumerate(rel[1:]):
                        arg = rel[0] + "Arg%i" % ai
                        if attr in X[i]:
                            tup += (X[i][attr],)
                            self.val_types[arg].add(X[i][attr])
                        else:
                            tup += (attr,)
                            self.val_types[arg].add(attr)

                    if not isinstance(X[i][rel], bool):
                        tup += (str(X[i][rel]),)
                        self.val_types[rel[0] + "Val"].add(str(X[i][rel]))

                    self.pos_aux_tuples[rel[0]].add(tup)

                else:
                    raise Exception("attribute not string or tuple.")



        # combine types and rename to combined types
        combined_types = {}
        overlap = {}
        for tt1 in self.val_types:
            for tt2 in self.val_types:
                if tt1 not in overlap:
                    overlap[tt1] = set()
                if tt1 == tt2: 
                    continue
                if (tt1 == self.example_type or tt2 == self.example_type):
                    continue
                if (len(self.val_types[tt1].intersection(self.val_types[tt2]))
                    > 0):
                    overlap[tt1].add(tt2)

        #pprint(overlap)
        num_types = 0

        open_list = set(overlap)
        closed_list = set()

        while len(open_list) > 0:
            inner_list = set()
            new_t = "type%i" % num_types
            num_types += 1
            combined_types[new_t] = set()
            inner_list.add(open_list.pop())

            while len(inner_list) > 0:
                curr_t = inner_list.pop()
                combined_types[new_t].add(curr_t)
                closed_list.add(curr_t)
                for some_t in overlap[curr_t]:
                    if some_t not in closed_list:
                        inner_list.add(some_t)
            open_list = open_list - closed_list

        #pprint(combined_types)
        self.type_mapping = {} 
        for tt in self.val_types:
            for ht in combined_types:
                if tt in combined_types[ht]:
                    self.type_mapping[tt] = ht
                    break
        #pprint(self.type_mapping)

        # create the background file for aleph
        data = ""
        
        # target relation mode
        data += ":- modeh(*," + self.name + "(" + ",".join(["+" + self.type_mapping[tt] for tt
                                            in self.target_types]) + ")).\n" 

        # aux relation modes
        for rel in self.aux_relations:
            for vt in product('+-#', repeat=len(self.aux_relations[rel]) - 1):
                vt = ('+',) + vt
                data += ":- modeb(*," + rel + "(" + ",".join([vt[i] +
                                                              self.type_mapping[tt]
                                                              for i, tt in
                                                     enumerate(self.aux_relations[rel])]) + ")).\n"

        # eq relation mode
        #for tt in combined_types:
        #    if tt == self.type_mapping[self.example_type]: 
        #        # don't need dif relation for example types.
        #        continue

        #    data += ":- modeb(*,(+" + tt + ")=(" + "#" + tt + ")).\n"
        #    data += ":- modeb(*,(-" + tt + ")=(" + "#" + tt + ")).\n"

        # dif relation mode (i.e., not equal).
        for tt in combined_types:
            if tt == self.type_mapping[self.example_type]: 
                # don't need dif relation for example types.
                continue

            for vt in product('+-', repeat=2):
                data += ":- modeb(*,dif(" + vt[0] + tt + "," + vt[1] + tt + ")).\n"
            data += ":- modeb(*,dif(+" + tt + "," + "#" + tt + ")).\n"
            data += ":- modeb(*,dif(-" + tt + "," + "#" + tt + ")).\n"

        # aux relation determinations
        data += "\n"
        for rel in self.aux_relations:
            data += ":- determination(" + self.name + "/%i" % (len(self.target_types))
            data += "," + rel + "/%i" % (len(self.aux_relations[rel])) + ").\n"

        # eq relation determination
        #data += ":- determination(" + self.name + "/%i" % (len(self.target_types))
        #data += ",'='/2).\n"

        # dif relation determination (i.e., not equal).
        data += ":- determination(" + self.name + "/%i" % (len(self.target_types))
        data += ",dif/2).\n"

        # type information
        data += "\n"
        printed_types = set()
        for tt in self.val_types:
            if self.type_mapping[tt] in printed_types:
                continue

            vals = set()
            for sub_t in combined_types[self.type_mapping[tt]]:
                for v in self.val_types[sub_t]:
                    vals.add(v)
            for v in vals:
                data += self.type_mapping[tt] + "(" + v + ").\n"
            printed_types.add(self.type_mapping[tt])

        # aux relations background info
        data += "\n"
        for rel in self.aux_relations:
            for t in self.pos_aux_tuples[rel]:
                data += rel + "(" + ",".join(t) + ").\n"

        #write background data
        with open("aleph_data.b", "w") as f:
            f.write(data)
        print(data)
        self.background_data = data

        # write positive examples file
        data = ""
        for t in self.pos:
            data += self.name + "(" + ",".join(t) + ").\n"

        with open("aleph_data.f", "w") as f:
            f.write(data)
        print(data)
                
        # write positive examples file
        data = ""
        for t in self.neg:
            data += self.name + "(" + ",".join(t) + ").\n"

        with open("aleph_data.n", "w") as f:
            f.write(data)
        print(data)

        #p = Popen(['swipl', '-q'],
        #          stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        # timeout = 5 min
        p = pexpect.spawn('swipl -q', timeout=300,
                          echo=True, encoding="utf-8", maxread=1000000,
                          searchwindowsize=None)
        
        p.expect("\?- ")
        p.sendline("[library(aleph)].")
        
        p.expect("\?- ")
        p.sendline("read_all(aleph_data).")
        p.expect("true ")
        p.send('.')

        p.expect("\?- ")
        #p.sendline("set(nodes, 100000).")
        p.sendline("set(clauselength, 10).")

        #p.expect("\?- ")
        #p.sendline("set(evalfn, posonly).")
        #if len(self.neg) == 0:
        #    p.sendline("set(evalfn, posonly).")
        #else:

        p.expect("\?- ")
        p.sendline("induce.")

        p.expect("\?- ")
        print(p.before)
        p.sendline('write_rules("aleph_data.pl").')

        # wait for the data to be saved?
        p.expect("\?- ")

        with open("aleph_data.pl", 'r') as f:
            self.rules = f.read()

        print("LEARNED RULES")
        print(self.rules)

    def get_matches(self, X, constraints=None):

        if self.rules == "":
            return

        if constraints == None:
            constraints = []

        X = {self.clean(a): self.clean(X[a]) for a in X}

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

        data = ":- style_check(-singleton).\n"
        
        for tt in set(self.type_mapping.values()):
            data += ":- discontiguous %s/1.\n" % tt
        
        data += self.type_mapping[self.example_type] + "(" + self.example_name + ").\n"
        for tt in val_types:
            for val in val_types[tt]:
                if tt in self.type_mapping:
                    data += self.type_mapping[tt] + "(" + val + ").\n"
                else:
                    data += tt + "(" + val + ").\n"

        for tt in self.val_types:
            if "targetArg" in tt:
                for val in self.val_types[tt]:
                    data += self.type_mapping[tt] + "(" + val + ").\n"

        # add aux relations to logic program
        rel_dict = dict()
        for rel in X:
            if isinstance(rel, str):
                rel_str = rel + "(" + self.example_name
                if not isinstance(X[rel], bool):
                    rel_str += ", " + str(X[rel]) 
                rel_str += ").\n"

                if rel not in rel_dict:
                    rel_dict[rel] = set()
                rel_dict[rel].add(rel_str)

            elif isinstance(rel, tuple):
                rel_str = rel[0] + "(" + self.example_name + ", " 
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

        data += self.rules

        args = [chr(i + ord("B")) for i in
                         range(len(self.target_types) - 1)]
        all_args = [self.example_name] + args

        bind_args = ['A'] + args
        data += "bind_relation(" + ",".join(bind_args) + ") :- "
        bind_body = [self.type_mapping[tt] + "(" + bind_args[i] + ")" 
                     for i,tt in enumerate(self.target_types)]
        bind_body = [self.name + "(" + ",".join(['EX'] + args) + ")"] + bind_body

        # user provided constraints
        bind_body += constraints

        data += ", ".join(bind_body)

        data += ".\n" 

        print(data)

        outfile = open("binding_lp.pl", 'w')
        outfile.write(data)
        outfile.close()

        #p = PopenSpawn('swipl -q -s foil_binding_lp.pl', encoding="utf-8",
        #               maxread=100000)

        # timeout = 5 min
        p = pexpect.spawn('swipl -q -s binding_lp.pl', timeout=300,
                          echo=False, encoding="utf-8", maxread=1000000,
                          searchwindowsize=None)


        p.expect("\?- ")

        p.sendline("bind_relation(" + ", ".join(all_args) + ").")
        
        patterns = []
        for ai, arg in enumerate(args):
            patterns.append(r"(?P<" + "arg%i" % (ai) + ">[" +
                            r"".join(args) + "0-9]+)" + r" = " + 
                            r"(?P<" + "argval%i" % (ai) + r">\w+)")
        arg_pattern = r",[\r\n ]+".join(patterns)

        #print(arg_pattern)

        mid_pattern = arg_pattern + r" "
        end_pattern = arg_pattern + r"."

        previous_matches = set()

        resp = -1
        while resp != 0:
            if len(args) > 0:
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
                mapping = {p.match.group('arg%i' % i): 
                           p.match.group('argval%i' % i) 
                           for i in range(len(args))}

                m = tuple([self.unclean(self.get_val(a, mapping)) 
                             for a in args])
                if m not in previous_matches:
                    yield m
                    previous_matches.add(m)

                if resp == 3:
                    p.send(";")
                else:
                    return


    def get_val(self, arg, mapping):
        if arg not in mapping:
            print(arg)
            print(mapping)
            raise Exception("arg not in mapping")
        if mapping[arg] in mapping:
            return self.get_val(mapping[arg], mapping)
        else:
            return mapping[arg]

    def unclean(self, x):
        if isinstance(x, tuple):
            return tuple(self.unclean(ele) for ele in x)
        elif isinstance(x, str):
            x = x.replace("multsign", "*")
            x = x.replace("equalsign", "=")
            x = x.replace("underscore", "_")
            x = x.replace("minussign", "-")
            x = x.replace("plussign", "+")
            x = x.replace("space", " ")
            x = x.replace("questionmark", "?")
            x = x[1:]
            if x == "nil":
                x == ""
            return x
        else:
            return x

    def clean(self, x):
        if isinstance(x, tuple):
            return tuple(self.clean(ele) for ele in x)
        elif isinstance(x, str):
            if x == "":
                x = "nil"
            x = x.replace("?", "questionmark")
            x = x.replace(" ", "space")
            x = x.replace("+", "plussign")
            x = x.replace("-", "minussign")
            x = x.replace("_", "underscore")
            x = x.replace("=", "equalsign")
            x = x.replace("*", "multsign")
            x = "a" + x
            return x
        else:
            return x
