PLANNERS = {}


class BasePlanner(object):
	def how_search(self,state,sai):
		raise NotImplementedError()
	def apply_featureset(self,state):
		raise NotImplementedError()
	def eval_expression(self,x,mapping,state):
		raise NotImplementedError()


def get_planner(name, **learner_kwargs):
	print(PLANNERS.keys())
	return PLANNERS[name.lower().replace(' ', '').replace('_', '')](**learner_kwargs)


# from planners.VectorizedPlanner import VectorizedPlanner
# from planners.fo_planner import FoPlanner
