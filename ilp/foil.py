from pprint import pprint
from numbers import Number
from subprocess import Popen
from subprocess import PIPE
from subprocess import STDOUT
import re
from copy import deepcopy

import pexpect
from pexpect.popen_spawn import PopenSpawn

from ilp.base_ilp import BaseILP


class Foil(BaseILP):

    def __init__(self, closed_world=False, max_tuples=100000):
        """
        The constructor. 
        """
        # Keywords used for constructing foil file (try not to conflict).
        self.name = "target_relation"
        self.example_type = "E"
        self.example_name = self.clean("curr_example")
        self.example_keyword = "e"

        # whether or not to use closed world assumption
        self.closed_world = closed_world
        self.max_tuples = max_tuples

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
        self.rules = set()

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

            ## Get all types and values
            #for rel in X[i]:
            #    if (not isinstance(X[i][rel], bool) and not
            #          isinstance(X[i][rel], Number)):
            #        # if not a boolean or number than add appropriate 
            #        # attr-val type info
            #        if isinstance(rel, str):
            #            if rel not in self.val_types:
            #                self.val_types[rel] = set()
            #            self.val_types[rel].add(X[i][rel])
            #        elif isinstance(rel, tuple):
            #            if rel[0] not in self.val_types:
            #                self.val_types[rel[0]] = set()
            #            self.val_types[rel[0]].add(X[i][rel])
            #        else:
            #            raise Exception("attribute not string or tuple.")
            #

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
                    if isinstance(X[i][rel], Number):
                        self.val_types[arg] = "continuous"
                    elif not isinstance(X[i][rel], bool):
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

        if self.closed_world is True:
            # only need to do sophisticated sampling when using closed world.
            num_neg_tuples = 1.0
            for t in self.target_types:
                num_neg_tuples *= len(self.val_types[t])

            num_pos_tuples = len(self.pos)
            max_neg_tuples = self.max_tuples - num_pos_tuples

            sample_size = min(10000, 10000 * ((max_neg_tuples-1) / num_neg_tuples))
            sample_size = int(sample_size) / 100

            print("NUM_NEG: ", num_neg_tuples)
            print("MAX_NEG: ", max_neg_tuples)
            print("SAMPLE: ", sample_size)

            # Create subprocess
            p = pexpect.spawn('ilp/FOIL6/foil6 -m %i -s %0.2f' %
                              (self.max_tuples, sample_size))
            p = Popen(['ilp/FOIL6/foil6', '-m %i' % self.max_tuples, '-s %0.2f' %
                       sample_size], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        else:
            p = Popen(['ilp/FOIL6/foil6', '-m %i' % self.max_tuples],
                      stdout=PIPE, stdin=PIPE, stderr=STDOUT)

        data = ""

        for tt in self.val_types:
            if tt == self.example_type:
                data += "#type" + tt + ": " + ",".join(["%s" % v for v in
                                                    self.val_types[tt]]) + ".\n"
            elif self.val_types[tt] == "continuous":
                data += "#type" + tt + ": " + self.val_types[tt] + ".\n"
            else:
                data += "#type" + tt + ": " + ",".join(["*%s" % v for v in
                                                    self.val_types[tt]]) + ".\n"
        data += "\n"

        # target relation header
        data += self.name + "(" + ", ".join(["type" + tt for tt in
                                             self.target_types]) + ") " 
        data += "".join(["#" for i in range(len(self.target_types))]) + "\n"

        # positive examples
        for t in self.pos:
            data += ", ".join(t) + "\n"

        if self.closed_world is False:
            # negative examples
            data += ";\n"
            for t in self.neg:
                data += ", ".join(t) + "\n"

        data += ".\n"

        # aux relations
        for rel in self.aux_relations:
            data += "*" + rel + "(" + ", ".join(['type' + tt for tt in
                                                 self.aux_relations[rel]]) + ")\n"

            # the positive examples of relation
            for t in self.pos_aux_tuples[rel]:
                data += ", ".join(t) + "\n"

            data += ".\n"

        print(data)

        # Send data to subprocess
        output = p.communicate(input=data.encode("utf-8"))[0]

        self.rules = set()

        found = False
            
        # Process result
        for line in output.decode().split("\n"):
            print(line)
            if re.search("^Training Set Size will exceed tuple limit:", line):
                raise Exception("Tuple limit exceeded, should never happen.")
            #match = re.search('^' + self.name + ".+:-", line)
            matches = re.findall("^" + self.name + "\(" +
                                 ",".join(["(?P<arg%i>[^ ,()]+)" % i for i in
                                           range(len(self.target_types))]) + "\)", line)
            if matches:
                found = True
                rule = ""
                args = matches[0]
                rule += re.sub("_[0-9]+", "_", 
                                     re.sub("not\((?P<content>[^)]+)\)", 
                                            "not \g<content>", line[:-1]))
                
                first_arg = False
                if " :- " not in line:
                    rule += " :- "
                    first_arg = True

                ground_atoms = set()
                for tt in self.val_types:
                    if self.val_types[tt] != "continuous":
                        ground_atoms = ground_atoms.union(self.val_types[tt])

                for i, arg in enumerate(args):
                    if arg not in ground_atoms:
                        if not first_arg:
                            rule += ", "
                        else:
                            first_arg = False
                        rule += "type" + self.target_types[i] + "(" + arg + ")"

                self.rules.add(rule)
                #self.rules += ".\n"
        
        #print("POS: ", len(self.pos))
        #print("NEG: ", len(self.neg))
        if not self.closed_world and not found and len(self.pos) > len(self.neg):
            rule = self.name + "(" + ",".join([chr(i + ord('A')) for i in range(len(self.target_types))]) + ") :- "
            rule += ", ".join(["type" + t + "(" + chr(i + ord('A')) + ")" for
                               i,t in enumerate(self.target_types)])
            self.rules.add(rule)
            #self.rules += ".\n"

        print("LEARNED RULES")
        print(self.rules)

    def get_matches(self, X):

        if self.rules == "":
            return

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

        #p = PopenSpawn('swipl -q', encoding="utf-8", maxread=100000)
        p = pexpect.spawn('swipl -q', timeout=5, echo=False, encoding="utf-8",
                          maxread=1000000, searchwindowsize=100)

        # add type info to logic program
        #p.expect("\?- ")
        #p.sendline("assert(type" + self.example_type + "(" + self.example_name
        #           + ")).")
        data = "assert(type" + self.example_type + "(" + self.example_name + ")).\n"

        for tt in val_types:
            if val_types[tt] != "continuous":
                for val in val_types[tt]:
                    data += "assert(type" + tt + "(" + val + ")).\n"
                    #p.expect("\?- ")
                    #p.sendline("assert(type" + tt + "(" + val + ")).")

        for tt in self.val_types:
            if "targetArg" in tt:
                if self.val_types[tt] != "continuous":
                    for val in self.val_types[tt]:
                        data += "assert(type" + tt + "(" + val + ")).\n"
                        #p.expect("\?- ")
                        #p.sendline("assert(type" + tt + "(" + val + ")).")

        # add aux relations to logic program
        for rel in X:
            if isinstance(rel, str):
                data += "assert(" + rel + "(" + self.example_name
                #data = "assert(" + rel + "(" + self.example_name
                if not isinstance(X[rel], bool):
                    data += ", " + str(X[rel]) 
                data += ")).\n"
                #p.expect("\?- ")
                #p.sendline(data)

            elif isinstance(rel, tuple):
                data += "assert(" + rel[0] + "(" + self.example_name + ", " 
                #data = "assert(" + rel[0] + "(" + self.example_name + ", " 
                vals = [X[attr] if attr in X else attr for attr in rel[1:]]
                data += ", ".join(vals)
                if not isinstance(X[rel], bool):
                    data += ", " + str(X[rel]) 
                data += ")).\n"
                #p.expect("\?- ")
                #p.sendline(data)

            else:
                raise Exception("attribute not string or tuple.")

        for rule in self.rules:
            # replace with appropriate prolog operators.
            rule = re.sub(" (?P<first>\w+)<>(?P<second>\w+)", 
                          " dif(\g<first>, \g<second>)", rule)
            rule = rule.replace("<=", "=<")
            data += "assert((" + rule + ")).\n"
            #p.expect("\?- ")
            #p.sendline("assert((" + rule + ")).")

        args = [chr(i + ord("B")) for i in
                         range(len(self.target_types) - 1)]
        all_args = [self.example_name] + args

        print(data)

        print("ready to send data")

        p.write(data)

        #p.sendline(data)

        print("data sent")

        p.sendline("print(expectme).")
        p.expect("expectme")

        p.expect("\?- ")

        p.sendline(self.name + "(" + ", ".join(all_args) + ").")
        
        patterns = []
        for arg in args:
            patterns.append(arg + " = " + "(?P<" + arg + r">\w+)")
        arg_pattern = ",\r\n".join(patterns)

        mid_pattern = arg_pattern + " "
        end_pattern = arg_pattern + "."

        resp = 1
        while resp != 0:
            if len(args) > 0:
                try:
                    resp = p.expect(["false.", "true.", mid_pattern, end_pattern])
                except Exception as e:
                    print(e)
                    print(p.buffer)
                    print(p.before)
            else:
                resp = p.expect(["false.", "true."])
            if resp == 1:
                yield tuple()
                break
            elif resp == 2 or resp == 3:
                mapping = {arg: p.match.group(arg) for arg in args}
                yield tuple([self.unclean(self.get_val(a, mapping)) 
                             for a in args])
                if resp == 2:
                    p.send(";")
                else:
                    break

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
