
from numbert.core import * #NBRT_KnowledgeBase, BaseOperator, Add, Subtract, 
from apprentice.planners.base_planner import BasePlanner, PLANNERS
from copy import deepcopy
import itertools
import logging
logging.getLogger("numba.core.ssa").setLevel(logging.ERROR)
logging.getLogger("numba.core.byteflow").setLevel(logging.ERROR)
logging.getLogger("numba.core.interpreter").setLevel(logging.ERROR)

def toFloatIfFloat(x):
	try:
		x = float(x)
	except ValueError:
		pass
	return x

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
		pass
		return state


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
		state = state.get_view("flat_ungrounded")

		if(search_depth == None): search_depth = self.search_depth
		if(operators == None): operators = self.function_set

		kb, back_map = state_as_kb(state,foci_of_attention)
		
		#Try to find a solution by looking for a number, if that doesn't work treat as string				
		operator_compositions = kb.how_search(operators,goal,search_depth=search_depth,max_solutions=100)
		print(operator_compositions)
		if(len(operator_compositions) == 0 and isinstance(goal,(int,float,bool))):
			operator_compositions = kb.how_search(operators,str_preserve_ints(goal),search_depth=search_depth,max_solutions=10)
		out = []

		at_least_one = False
		for op_comp in operator_compositions:
			mappings = []
			arg_sources = [back_map[arg.binding] for i,arg in enumerate(op_comp.args)]
			arg_combinations = itertools.product(*arg_sources)

			if(not allow_copy and op_comp.depth == 0 and min([op.depth for op in operators]) > 0):
				continue	

			op_comp = deepcopy(op_comp)
			op_comp.unbind()
			if(out_type == 'str'):
				op_comp.force_cast('string')
			for args in arg_combinations:
				
				if(len(set(args)) != len(args)): continue
				# print("args1",args)
				# if(list(args) != sorted(args)): continue #Note to self... This is probably not robust
				if(foci_of_attention != None and len(foci_of_attention) != len(args)): continue				
				# print("args",args)

				mapping = {"?arg%s"%i:arg[1] for i,arg in enumerate(args)}
				at_least_one = True
				yield op_comp, mapping

		if(not at_least_one and allow_bottomout and 
			(foci_of_attention == None or len(foci_of_attention) == 0)):
			
			yield goal, {}


	def eval_expression(self,expr,mapping,state):
		if(isinstance(expr,list)):
			return [self.eval_expression(e,mapping,state) for e in expr]
		else:
			if(not isinstance(expr,OperatorComposition)):
				return expr
			state = state.get_view("flat_ungrounded")
			literals = []
			for arg_str,literal_src in mapping.items():
				if("sel" in arg_str):continue
				val = state[('value',literal_src)]
					
				literals.append(toFloatIfFloat(val))
			return expr(*literals)


	def unify_op(self,state,op,sai,foci_of_attention=None):
		if(foci_of_attention != None and len(foci_of_attention) != len(op.args)):
			return []						
		out_type = type(sai.inputs["value"]).__name__
		goal = toFloatIfFloat(sai.inputs["value"])
		state = state.get_view("flat_ungrounded")
		kb, back_map = state_as_kb(state,foci_of_attention=foci_of_attention)
		arg_val_sets = kb.unify_op(op,goal)
		if(len(arg_val_sets) == 0):
			arg_val_sets = kb.unify_op(op,str_preserve_ints(goal))
		mappings = []
		for arg_val_set in arg_val_sets:
			arg_sources = [back_map[arg] for i,arg in enumerate(arg_val_set)]
			arg_combinations = itertools.product(*arg_sources)
			for args in arg_combinations:
				#Don't allow redundancy
				# if(list(args) != sorted(args)): continue #Note to self... This is probably not robust
				if(len(set(args)) != len(args)): continue				

				mappings.append({"?arg%s"%i:arg[1] for i,arg in enumerate(args)})
		return mappings




#Putting these tests in main for now, because much of this class might be deprecated soon
#	if I figure out using the planner with objects + backward chaining
if __name__ == "__main__":
	from apprentice.working_memory.representation.representation import StateMultiView
	state = {('value','crabman') : 5,
			 ('value','lobsterman') : 3,
			 ('value','lobsterman2') : 3,
			 ('value','whalefriend') : "WHALE!",
			 ('value','merman') : "7",
			 # ('value','solution') : "15",
			 ('value','mermaid') : "",
			 }
	state = StateMultiView("flat_ungrounded", state)
	class SAI(object):
		pass
	sai = SAI()
	sai.selection = 'mermaid'
	sai.action = 'befriend'
	sai.inputs = {'value': '15'}


	# {'selection' : , 'action': 'befriend', 'inputs' : {'value': '12'}}
	planner = NumbaPlanner(search_depth=2,function_set=[StrToFloat, Add,Subtract],feature_set=[Add,Subtract])
	# print("BEFORE")
	out = planner.how_search(state,sai)
	# print(list(out))
	for expr,mapping in out:
		print("EXPR",expr, expr.args, expr.out_type, expr.arg_types, mapping.values())
		evaled = planner.eval_expression(expr,mapping,state)
		# print(mapping,evaled,type(evaled).__name__)
		assert evaled == sai.inputs['value'], "%s != %s" % (evaled, sai.inputs['value'])
		assert expr.depth == 2
		mappings = planner.unify_op(state,expr,sai) 
		# print("SHEEE",mappings)
		assert len(mappings) > 0
		# print(v)
	# print("OUT",out)
	# raise ValueError()
	# assert 


	state = {('value','thing') : 7,
			 ('value','same_thing') : "7",
			 ('value','same_thing2') : "7",
			 ('value','empty_thing') : "",
			 }
	state = StateMultiView("flat_ungrounded", state)
	sai = SAI()
	sai.selection = 'empty_thing'
	sai.action = 'something'
	sai.inputs = {'value': '14'}

	#Test Search
	planner = NumbaPlanner(search_depth=1,function_set=[StrToFloat, Add,Subtract],feature_set=[Add,Subtract])
	# print("BEFORE")
	out = planner.how_search(state,sai)
	for expr,mapping in out:
		print("EXPR",expr, expr.args, expr.out_type, expr.arg_types, mapping.values())
		evaled = planner.eval_expression(expr,mapping,state)
		# print(expr,mapping,evaled,type(evaled).__name__)
		assert evaled == sai.inputs['value'], "%s != %s" % (evaled, sai.inputs['value'])
		assert expr.depth == 1

	# print("START UNIFY")

	# kb, back_map = state_as_kb(state.get_view("flat_ungrounded"))
	assert len(planner.unify_op(state,Add,sai)) > 0
	# print("END UNIFY")


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
	planner = NumbaPlanner(search_depth=1,function_set=[],feature_set=[])
	out = [x for x in planner.how_search(state,sai)]
	
	for expr,mapping in out:
		evaled = planner.eval_expression(expr,mapping,state)
		# print("Copy1",expr.tup,type(expr.tup),expr.template,mapping)
		assert evaled == sai.inputs['value'], "%s != %s" % (evaled, sai.inputs['value'])
		assert expr.depth == 0
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


	print("BRK")
	#Test Solutions at Multiple Depths
	state = {('value','thing') : 7,
			 ('value','same_thing') : "7",
			 ('value','same_thing2') : "7",
			 ('value','empty_thing') : "",
			 ('value','target_thing') : "21",
			 }
	state = StateMultiView("flat_ungrounded", state)
	sai = SAI()
	sai.selection = 'empty_thing'
	sai.action = 'something'
	sai.inputs = {'value': '21'}
	planner = NumbaPlanner(search_depth=2,function_set=[Add],feature_set=[],)
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
	assert 0 in depths and 2 in depths
	# assert len(out) == 3


	shloop = OperatorComposition((Add3 ,Var(),(Div10,Var()),(Div10,Var())))
	print(len(shloop.args))
	shloop(1,2,3)


	#Test Multi-Column
	state = {('value','A') : "1",
			 ('value','B') : "7",
			 ('value','C') : "7",
			 ('value','out') : "",
			 }
	state = StateMultiView("flat_ungrounded", state)
	sai = SAI()
	sai.selection = 'out'
	sai.action = 'something'
	sai.inputs = {'value': '1'}
	planner = NumbaPlanner(search_depth=2,function_set=[Div10,Mod10,Add3,Add],feature_set=[])
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
	# assert len(depths) == 2
	# assert 0 in depths and 2 in depths
	# assert len(out) == 3


	# broadcast_forward_op_comp(kb,expr)
	print("ALL TESTS PASSED")

PLANNERS["numbert"] = NumbaPlanner
PLANNERS["numba"] = NumbaPlanner