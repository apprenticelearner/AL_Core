PLANNERS = {}


class BasePlanner(object):
	def how_search(self,state,sai):
		raise NotImplementedError()
	def apply_featureset(self,state):
		raise NotImplementedError()
	def eval_expression(self,x,mapping,state):
		raise NotImplementedError()
	def resolve_operators(operators):
		raise NotImplementedError()


def get_planner_class(name):
	if(name == "vectorized"):
		from planners.VectorizedPlanner import VectorizedPlanner
	name = name.lower().replace(' ', '').replace('_', '')
	return PLANNERS[name]


# from planners.VectorizedPlanner import VectorizedPlanner
# from planners.fo_planner import FoPlanner
