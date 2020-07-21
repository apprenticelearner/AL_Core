import logging
logging.getLogger("numba.core.ssa").setLevel(logging.ERROR)
logging.getLogger("numba.core.byteflow").setLevel(logging.ERROR)
logging.getLogger("numba.core.interpreter").setLevel(logging.ERROR)

from numba import njit #NBRT_KnowledgeBase, BaseOperator, Add, Subtract, 
from numba.typed import Dict,List #NBRT_KnowledgeBase, BaseOperator, Add, Subtract, 
from numbert.knowledgebase import NBRT_KnowledgeBase #NBRT_KnowledgeBase, BaseOperator, Add, Subtract, 
from numbert.operator import BaseOperator, OperatorComposition, str_preserve_ints, Var
from apprentice.planners.base_planner import BasePlanner, PLANNERS
import apprentice.working_memory.numba_operators
from copy import deepcopy
import itertools
import math
import numpy as np


def toFloatIfFloat(x):
	try:
		x = float(x)
	except ValueError:
		pass
	return x

@njit(cache=True)
def get_nb_substate(state,ids):
	out = Dict()
	for k in ids:
		if(k in state):
			out[k] = state[k]
	return out

def state_as_kb2(state,foci_of_attention=None):
	nb_state = state.get_view('nb_object')
	if(foci_of_attention is not None):
		nb_foci = List(foci_of_attention)
		new_nb_state = {}
		for typ in nb_state:
			new_nb_state[typ] = get_nb_substate(nb_state[typ],nb_foci)
		nb_state = new_nb_state
	kb = NBRT_KnowledgeBase()
	# print(nb_state)
	kb.declare(nb_state)
	return kb


def state_as_kb(state,foci_of_attention=None):

	if(foci_of_attention != None):
		state = {k:v for k,v in state.items() if k[1].replace("?ele-","") in foci_of_attention}

	kb = NBRT_KnowledgeBase()

	back_map = {}
	for key, val in state.items():
		if(isinstance(key, tuple) and key[0] == 'value' and val != ""):
			val = toFloatIfFloat(val)
			arr = back_map[val] = back_map.get(val,[])
			arr.append(key)
			kb.declare(val)
	return kb, back_map

class NumbaPlanner(BasePlanner):
	def __init__(self,search_depth,function_set,feature_set,**kwargs):
		self.function_set = function_set
		self.feature_set = feature_set
		self.epsilon = 0.0
		self.search_depth = search_depth

	@classmethod
	def resolve_operators(cls,operators):
		out = []
		for op in operators:
			if(isinstance(op, BaseOperator)):
				out.append(op)
			elif(isinstance(op, str)):
				try:
					out.append(BaseOperator.registered_operators[op.lower()])
				except KeyError as e:
					raise KeyError("No Operator registered under name %s" % op) 
			else:
				raise ValueError("Cannot resolve operator of type %s" % type(op))
		return out

	def apply_featureset(self, state, operators=None):
		if(operators == None): operators = self.feature_set
		flat_state = state.get_view("flat_ungrounded")
		kb, back_map = state_as_kb(flat_state)

		applied_features = {}

		kb.forward(operators)
		for typ,hist in kb.hists.items():
			if(len(hist) > 1):
				for record in hist[1]:
					op_uid, _hist, shape, arg_types, vmap = record
					op_name = BaseOperator.operators_by_uid[op_uid].__name__
					hist_reshaped = _hist.reshape(shape)
					arg_sets = [kb.u_vs.get(t,[]) for t in arg_types]
					for v,uid in vmap.items():
						inds = np.stack(np.where(hist_reshaped == uid)).T
						for ind in inds:
							arg_set = [arg_sets[i][j] for i,j in enumerate(ind)]
							for args in itertools.product(*[back_map[a] for a in arg_set]):
								applied_features[(op_name,*args)] = v
		flat_state.update(applied_features)
		return flat_state


	def how_search(self,state,
					sai,
					operators=None,
					foci_of_attention=None,
					search_depth=None,
					allow_bottomout=True,
					allow_copy=True,
					epsilon=0.0):
		# print("HOW_SEARCH D=", search_depth, "N_ops=",len(operators) if operators is not None else -1)
		assert "value" in sai.inputs, "For now NumbaPlanner only searches for exaplantions of SAIs with inputs['value'] set."

		#Treat Button Presses Specially
		if(sai.action == "ButtonPressed"):
			if(operators == None):
				yield -1,{}
				return
			else:
				return


		out_type = type(sai.inputs["value"]).__name__
		goal = toFloatIfFloat(sai.inputs["value"])
		# state = state.get_view("flat_ungrounded")

		if(search_depth == None): search_depth = self.search_depth
		if(operators == None): operators = self.function_set

		kb = state_as_kb2(state,foci_of_attention)
		
		#Try to find a solution by looking for a number, if that doesn't work treat as string				
		operator_compositions = kb.how_search(operators,goal,search_depth=search_depth,max_solutions=100)
		# print(operator_compositions)
		if(len(operator_compositions) == 0 and isinstance(goal,(int,float,bool))):
			operator_compositions = kb.how_search(operators,str_preserve_ints(goal),search_depth=search_depth,max_solutions=100)
		out = []

		at_least_one = False
		for op_comp in operator_compositions:
			if(not allow_copy and op_comp.depth == 0 and min([op.depth for op in operators]) > 0):
				continue	

			op_comp = deepcopy(op_comp)
			args = [arg.binding.id for arg in op_comp.args]
			if(len(set(args)) != len(args)): continue
			if(foci_of_attention != None and len(foci_of_attention) != len(args)): continue				

			mapping = {"?arg%s"%i:arg for i,arg in enumerate(args)}
			op_comp.unbind()

			if(out_type == 'str'):
				op_comp.force_cast('string')
			at_least_one = True
			# print('HERE:',op_comp,args)
			yield op_comp, mapping

		if(not at_least_one and allow_bottomout and 
			(foci_of_attention == None or len(foci_of_attention) == 0)):
			# print("Bottomout!!!!!!!!!!!!!!!!!!!!!!!")
			yield goal, {}


	def eval_expression(self,expr,mapping,state):
		if(isinstance(expr,list)):
			return [self.eval_expression(e,mapping,state) for e in expr]
		else:
			if(not isinstance(expr,OperatorComposition)):
				return expr
			state = state.get_view("nb_object")

			ids = [v for k,v in mapping.items() if k != "?sel"]
			literals = [state[typ][_id] for typ, _id in zip(expr.arg_types,ids)]
			return expr(*literals)
			


	def unify_op(self,state,op,sai,foci_of_attention=None):
		if(foci_of_attention != None and len(foci_of_attention) != len(op.args)):
			return []						
		out_type = type(sai.inputs["value"]).__name__
		goal = toFloatIfFloat(sai.inputs["value"])
		# state = state.get_view("flat_ungrounded")
		kb = state_as_kb2(state,foci_of_attention=foci_of_attention)
		arg_val_sets = kb.unify_op(op,goal)
		if(len(arg_val_sets) == 0):
			arg_val_sets = kb.unify_op(op,str_preserve_ints(goal))
		mappings = []
		for arg_val_set in arg_val_sets:
			args = [arg.id for arg in arg_val_set]
			# arg_sources = [back_map[arg] for i,arg in enumerate(arg_val_set)]
			# arg_combinations = itertools.product(*arg_sources)
			# for args in arg_combinations:
				#Don't allow redundancy
				# if(list(args) != sorted(args)): continue #Note to self... This is probably not robust
			if(len(set(args)) != len(args)): continue				

			mappings.append({"?arg%s"%i:arg for i,arg in enumerate(args)})
		return mappings

def state_of_ies_from_dict(d):
	out = {}
	for k, v in d.items():
		out[k] = {'type': "TextField", "id": k, "value": v}
	return out


#Putting these tests in main for now, because much of this class might be deprecated soon
#	if I figure out using the planner with objects + backward chaining
if __name__ == "__main__":
	class SAI(object):
		pass

	from apprentice.working_memory.representation.representation import StateMultiView, numbalizer
	
	ie_spec = {
		"id" : "string",
		"value" : "string"
	}

	numbalizer.register_specification("TextField",ie_spec)

	class RipFloatValue(BaseOperator):
		signature = 'float(TextField)'
		template = "{}.v"
		nopython=False
		muted_exceptions = [ValueError]
		def forward(x): 
			return float(x.value)

	state = state_of_ies_from_dict(
				{"crabman" : 5,
				 "lobsterman" : 3,
				 "lobsterman2" : 3,
				 "whalefriend" : "WHALE!",
				 "merman" : "7",
				 "mermaid" : "",
				})		

	state = StateMultiView("object", state)
	planner = NumbaPlanner(search_depth=3,function_set=[RipFloatValue, Add,Subtract],feature_set=[])
	
	
	sai = SAI()
	sai.selection = 'mermaid'
	sai.action = 'befriend'
	sai.inputs = {'value': '15'}

	out = planner.how_search(state,sai)
	for expr,mapping in out:
		print("EXPR",expr, expr.args, expr.out_type, expr.arg_types, mapping.values())
		evaled = planner.eval_expression(expr,mapping,state)
		# print(mapping,evaled,type(evaled).__name__)
		assert evaled == sai.inputs['value'], "%s != %s" % (evaled, sai.inputs['value'])
		print("expr.depth", expr.depth)
		assert expr.depth == 3
		mappings = planner.unify_op(state,expr,sai) 
		# print("SHEEE",mappings)
		assert len(mappings) > 0
		# print(expr, mapping)


	state = state_of_ies_from_dict(
				{"thing" : 7,
				 "same_thing" : "7",
				 "same_thing2" : "7",
				 "empty_thing" : "",
				})		

	state = StateMultiView("object", state)
	sai = SAI()
	sai.selection = 'empty_thing'
	sai.action = 'something'
	sai.inputs = {'value': '14'}

	#Test Search
	planner = NumbaPlanner(search_depth=2,function_set=[RipFloatValue, Add,Subtract],feature_set=[Add,Subtract])
	# print("BEFORE")
	out = planner.how_search(state,sai)
	for expr,mapping in out:
		print("EXPR",expr, expr.args, expr.out_type, expr.arg_types, mapping.values())
		evaled = planner.eval_expression(expr,mapping,state)
		# print(expr,mapping,evaled,type(evaled).__name__)
		assert evaled == sai.inputs['value'], "%s != %s" % (evaled, sai.inputs['value'])
		assert expr.depth == 2

	#Note: should put test case somewhere of unifying pure-ops
	# assert len(planner.unify_op(state,Add,sai)) > 0

	

	#Test Search with OperatorComposition
	planner = NumbaPlanner(search_depth=1,function_set=[expr],feature_set=[Add,Subtract])
	out = [x for x in planner.how_search(state,sai,operators=[expr],search_depth=1)]
	assert len(out) > 0
	for expr,mapping in out:
		print("EXPR",expr, expr.args, expr.out_type, expr.arg_types, mapping.values())
		evaled = planner.eval_expression(expr,mapping,state)
		# print("OUT2",expr,mapping,evaled,type(evaled).__name__)
		assert evaled == sai.inputs['value'], "%s != %s" % (evaled, sai.inputs['value'])
		assert expr.depth == 1
		assert len(planner.unify_op(state,expr,sai)) > 0

	#Test Copy
	sai = SAI()
	sai.selection = 'empty_thing'
	sai.action = 'something'
	sai.inputs = {'value': 7}
	planner = NumbaPlanner(search_depth=1,function_set=[RipFloatValue],feature_set=[])
	out = [x for x in planner.how_search(state,sai)]
	print(out)
	for expr,mapping in out:
		evaled = planner.eval_expression(expr,mapping,state)
		# print("Copy1",expr.tup,type(expr.tup),expr.template,mapping)
		assert evaled == sai.inputs['value'], "%s != %s" % (evaled, sai.inputs['value'])
		assert expr.depth == 1
		assert len(planner.unify_op(state,expr,sai)) > 0
	assert len(out) == 3

	#Test Bottomout
	sai = SAI()
	sai.selection = 'empty_thing'
	sai.action = 'something'
	sai.inputs = {'value': 'SoME VAlue WHich DOEs NOt Exist'}
	out = [x for x in planner.how_search(state,sai)]
	assert len(out) == 1
	out = [x for x in planner.how_search(state,sai,allow_bottomout=False)]
	assert len(out) == 0

	# raise ValueError("DONE")
	print("BRK")

	#Test Solutions at Multiple Depths
	state = state_of_ies_from_dict(
				{"thing" : 7,
				 "same_thing" : "7",
				 "same_thing2" : "7",
				 "empty_thing" : "",
				 "target_thing" : "21",
				})		
	
	state = StateMultiView("object", state)
	sai = SAI()
	sai.selection = 'empty_thing'
	sai.action = 'something'
	sai.inputs = {'value': '21'}
	planner = NumbaPlanner(search_depth=3,function_set=[RipFloatValue, Add],feature_set=[],)
	out = [x for x in planner.how_search(state,sai)]
	
	depths = set()
	for expr,mapping in out:
		print("EXPR",expr, mapping)
		# print("EXPR",expr, expr.args, expr.out_type, expr.arg_types, mapping.values())
		evaled = planner.eval_expression(expr,mapping,state)
		# print("Copy1",expr.tup,type(expr.tup),expr.template,mapping)
		assert evaled == sai.inputs['value'], "%s != %s" % (evaled, sai.inputs['value'])
		# assert expr.depth == 0
		depths.add(expr.depth)
		assert len(planner.unify_op(state,expr,sai)) > 0
	assert len(depths) == 2
	assert 1 in depths and 3 in depths
	# assert len(out) == 3


	shloop = OperatorComposition((Add3 ,Var(),(Div10,Var()),(Div10,Var())))
	print(len(shloop.args))
	shloop(1,2,3)

	
	#Test Multi-Column
	state = state_of_ies_from_dict(
				{"A" : "1",
				 "B" : "7",
				 "C" : "7",
				 "out" : "",
				})		
	state = StateMultiView("object", state)
	sai = SAI()
	sai.selection = 'out'
	sai.action = 'something'
	sai.inputs = {'value': '1'}
	planner = NumbaPlanner(search_depth=3,function_set=[RipFloatValue,Div10,Mod10,Add3,Add],feature_set=[])
	out = [x for x in planner.how_search(state,sai,foci_of_attention=['A','B','C'])]
	
	depths = set()
	for expr,mapping in out:
		print("EXPR",expr, mapping)
		# print("EXPR",expr, expr.args, expr.out_type, expr.arg_types, mapping.values())
		evaled = planner.eval_expression(expr,mapping,state)
		# print("Copy1",expr.tup,type(expr.tup),expr.template,mapping)
		assert evaled == sai.inputs['value'], "%s != %s" % (evaled, sai.inputs['value'])
		# assert expr.depth == 0
		depths.add(expr.depth)
		assert len(planner.unify_op(state,expr,sai)) > 0

	# raise ValueError("DONE")
	# assert len(depths) == 2
	# assert 0 in depths and 2 in depths
	# assert len(out) == 3
	#Test Apply Featureset
	planner.apply_featureset(state,[Equals])
	print(state.get_view("flat_ungrounded"))
	flat_state = state.get_view("flat_ungrounded")
	assert flat_state[('Equals', ('value', 'B'), ('value', 'B'))] ==  1.0
	assert flat_state[('Equals', ('value', 'A'), ('value', 'B'))] ==  0.0
	assert flat_state[('Equals', ('value', 'B'), ('value', 'C'))] ==  1.0


	# broadcast_forward_op_comp(kb,expr)
	print("ALL TESTS PASSED")

PLANNERS["numbert"] = NumbaPlanner
PLANNERS["numba"] = NumbaPlanner