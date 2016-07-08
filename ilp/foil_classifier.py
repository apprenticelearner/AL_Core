from pprint import pprint
from numbers import Number
from subprocess import Popen
from subprocess import PIPE
from subprocess import STDOUT
import re

import pexpect

from ilp.base_ilp import BaseILP

class FoilClassifier(BaseILP):

    def __init__(self, closed_world=True, max_tuples=100000,
                 name="target_relation"):
        """
        The constructor. 
        """
        # Keywords used for constructing foil file (try not to conflict).
        self.name = name

        # whether or not to use closed world assumption
        self.closed_world = closed_world
        self.max_tuples = max_tuples

        # initialize all class variables
        self.initialize()

    def initialize(self):

        # tuples for target relation
        self.pos = list()
        self.neg = list()

        # the possible attr value types
        # the val type "continuous" is reserved for numerical types
        self.target_types = []
        self.val_types = {}

        # the aux relations and their type tuples
        # also the pos tuples for aux relations
        self.aux_relations = {}
        self.pos_aux_tuples = {}

        # String representation of learned rules
        self.rules = set()

    def fit(self, X, y):

        # initialize all variables for learning
        self.initialize()

        # clean the input
        X = [{self.clean(a): self.clean(x[a]) for a in x} for x in X]

        for x in X:
            for attr in x:
                if not isinstance(attr, tuple):
                    if attr not in self.val_types:
                        self.val_types[attr] = set()
                    if isinstance(x[attr], Number):
                        self.val_types[attr].add('continuous')
                    else:
                        self.val_types[attr].add("*" + x[attr])

        self.target_types = list(set(self.val_types.keys()))
        self.target_types.sort()

        # do something to find auxillery relations

        for i, correct in enumerate(y):

            e = tuple([str(X[i][attr]) if attr in X[i] else '?' for attr in
                       self.target_types])
            #print(e)

            # Get target relation tuples
            if correct == 1:
                self.pos.append(e)
            else:
                self.neg.append(e)

            # do something to collect auxillary relations info

        data = ""

        for tt in self.val_types:
            data += "#" + tt + ": " + ", ".join(self.val_types[tt]) + ".\n"

        data += "\n"
        # rename target relation header to new types

        # target relation header
        data += self.name + "(" + ", ".join(self.target_types) + ")\n" 

        # positive examples
        for t in self.pos:
            data += ", ".join(t) + "\n"

        if self.closed_world is False:
            # negative examples
            data += ";\n"
            for t in self.neg:
                data += ", ".join(t) + "\n"

        data += ".\n"

        # TODO add the aux relation info
        # aux relations
        #for rel in self.aux_relations:
        #    data += "*" + rel + "(" + ", ".join([self.type_mapping[tt] for tt in
        #                                         self.aux_relations[rel]]) + ")\n"

        #    # the positive examples of relation
        #    for t in self.pos_aux_tuples[rel]:
        #        data += ", ".join(t) + "\n"

        #    data += ".\n"

        print(data)

        if self.closed_world is True:
            # only need to do sophisticated sampling when using closed world.
            num_neg_tuples = 1.0
            for t in self.target_types:
                vals = set()
                for v in self.val_types[t]:
                    if v != "continuous":
                        vals.add(v)
                num_neg_tuples *= len(vals)

            num_pos_tuples = len(self.pos)
            max_neg_tuples = self.max_tuples - num_pos_tuples

            sample_size = min(10000, 10000 * ((max_neg_tuples-1) / num_neg_tuples))
            sample_size = int(sample_size) / 100

            #print("NUM_NEG: ", num_neg_tuples)
            #print("MAX_NEG: ", max_neg_tuples)
            #print("SAMPLE: ", sample_size)

            # Create subprocess
            #p = pexpect.spawn('ilp/FOIL6/foil6 -m %i -s %0.2f -a %0.2f' %
            #                  (self.max_tuples, sample_size, 0.6))
            p = Popen(['ilp/FOIL6/foil6', '-m %i' % self.max_tuples, '-s %0.2f' %
                       sample_size], stdout=PIPE, stdin=PIPE, stderr=STDOUT)
        else:
            p = Popen(['ilp/FOIL6/foil6', '-m %i' % self.max_tuples],
                      stdout=PIPE, stdin=PIPE, stderr=STDOUT)

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
                rule = line[:-1]
                #rule += re.sub("_[0-9]+", "_", 
                #                     re.sub("not\((?P<content>[^)]+)\)", 
                #                            "not \g<content>", line[:-1]))
                
                self.rules.add(rule)
                
        
        #print("POS: ", len(self.pos))
        #print("NEG: ", len(self.neg))
        if self.closed_world is False and not found and len(self.pos) > len(self.neg):
            rule = self.name 
            rule += "(" + ",".join([chr(i + ord('A')) for i in
                                    range(len(self.target_types))]) + ") :- "
            rule += ", ".join([t + "(" + chr(i + ord('A')) + ")" 
                               for t in self.target_types])
            self.rules.add(rule)
            #self.rules += ".\n"

        print("LEARNED RULES")
        print(self.rules)

    def predict(self, X):

        if len(self.rules) == 0:
            return [0 for x in X]

        # maybe update aux relation here

        data = ""

        #for tt in self.val_types:
        #    if self.val_types[tt] != "continuous":
        #        for val in self.val_types[tt]:
        #            data += tt + "(" + val[1:] + ").\n"

                    #p.expect("\?- ")
                    #p.sendline("assert(type" + tt + "(" + val + ")).")

        # add aux relations to logic program.

        for rule in self.rules:
            # replace with appropriate prolog operators.
            rule = re.sub(" (?P<first>\w+)<>(?P<second>\w+)", 
                          " dif(\g<first>, \g<second>)", rule)
            #rule = re.sub(" not (?P<term>[\w,()]+),", 
            #              " not(\g<term>),", rule)
            rule = rule.replace("<=", "=<")
            data += rule + ".\n"

        #args = [chr(i + ord("A")) for i in
        #                 range(len(self.target_types))]

        #bind_args = ['A'] + args
        #data += "bind_relation(" + ",".join(bind_args) + ") :- "
        #bind_body = [self.type_mapping[tt] + "(" + bind_args[i] + ")" 
        #             for i,tt in enumerate(self.target_types)]
        #bind_body = [self.name + "(" + ",".join(['A'] + args) + ")"] + bind_body

        ## user provided constraints
        #bind_body += constraints
        #data += ", ".join(bind_body)
        #data += ".\n" 

        print(data)

        #print(data)

        outfile = open("foil_binding_lp.pl", 'w')
        outfile.write(data)
        outfile.close()

        #p = PopenSpawn('swipl -q -s foil_binding_lp.pl', encoding="utf-8",
        #               maxread=100000)
        p = pexpect.spawn('swipl -q -s foil_binding_lp.pl', timeout=None,
                          echo=False, encoding="utf-8", maxread=1000000,
                          searchwindowsize=None)
        p.expect("\?- ")

        X = [{self.clean(a): self.clean(x[a]) for a in x} for x in X]

        yh = []
        for x in X:
            e = tuple([str(x[attr]) if attr in x else '_' for attr in
                       self.target_types])

            p.sendline(self.name + "(" + ", ".join(e) + ").")
            resp = p.expect(["false.", "true."])

            if resp == 0:
                yh.append(0)
            elif resp == 1:
                yh.append(1)

        return yh

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
            x = x.replace("leftparen", '(')
            x = x.replace("rightparen", ')')
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
            x = x.replace("(", "leftparen")
            x = x.replace(")", "rightparen")
            x = "a" + x
            return x
        else:
            return x
