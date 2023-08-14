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
	def unify_op(self,state,op,sai,foci_of_attention=None):
		raise NotImplementedError()


def get_planner_class(name):
	print(type(name))
	# if is class then return
	if(isinstance(name, type)):
		return name
	if(name == "vectorized"):
		from apprentice.planners.VectorizedPlanner import VectorizedPlanner
	name = name.lower().replace(' ', '').replace('_', '')
	return PLANNERS[name]


# from planners.VectorizedPlanner import VectorizedPlanner
# from planners.fo_planner import FoPlanner
