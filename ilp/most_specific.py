from ilp.base_ilp import BaseILP

class MostSpecific(BaseILP):

    def __init__(self):
        self.pos = set()
        self.target_types = []

    def get_matches(self, X, constraints=None):
        for t in self.pos:
            yield t
        
    def fit(self, T, X, y):

        self.target_types = T[0]

        #ignore X and save the positive T's.
        for i,t in enumerate(T):
            if y[i] == 1:
                self.pos.add(t)

    def __repr__(self):
        tt = []
        try:
            for t in self.pos:
                t = ["('name', '?foa%i')=='%s'" % (i,v) for i,v in enumerate(t)]
                tt.append(t)
        except:
            pass

        tt = ["(" + ", ".join(t) + ")" for t in tt]
        return repr(" or ".join(tt))
