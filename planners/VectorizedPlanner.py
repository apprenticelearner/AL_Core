if __name__ == "__main__":
	import sys
	sys.path.insert(0,"../")

import torch

import itertools
import re 
from torch.nn import functional as F

from concept_formation.preprocessor import Flattener
from concept_formation.preprocessor import Tuplizer
from planners.base_planner import BasePlanner, PLANNERS

function_set = []
feature_set = []

class BaseOperator(object):
	def __init__(self):
		self.num_flt_inputs = 0;
		self.num_str_inputs = 0;
		self.out_arg_types = ["value"]
		self.in_arg_types= ["value","value"]
		self.commutative = False;
		self.template = "BaseOperator"
	

	def forward(self, args):
		raise NotImplementedError("Not Implemeneted")

	def backward(self, args):
		raise NotImplementedError("Not Implemeneted")
	def search_mask(self,*args):
		return args
	def __str__(self):
		# print(self.template)
		# print(self.in_arg_types)
		# print(tuple(["E" + str(i) for i in range(len(self.in_arg_types))]))
		return self.template.format(*["E" + str(i) for i in range(len(self.in_arg_types))])



NaN = torch.tensor(float("NaN"))

class Add(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 2;
		self.template = "({} + {})"
		self.commutative = True;

	def forward(self, x,y):
		return x+y

class Add3(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 3;
		self.in_arg_types= ["value","value","value"]
		self.template = "({} + {} + {})"
		self.commutative = True;

	def forward(self, x,y,z):
		return x+y+z

class Subtract(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 2;
		self.template = "({} - {})"

	def forward(self, x,y):
		return x-y

class Multiply(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 2;
		self.template = "({} * {})"
		self.commutative = True;

	def forward(self, x,y):
		return x*y

class Divide(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 2;
		self.template = "({} / {})"

	def forward(self, x,y):
		return x/y

class Mod10(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 1;
		self.template = "({} % 10)"
		self.in_arg_types = ["value"]

	def forward(self, x):
		return x % 10

	def search_mask(self,x):
		return [torch.where(x >= 10, x, NaN)]

class Div10(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 1;
		self.template = "({} // 10)"
		self.in_arg_types = ["value"]

	def forward(self, x):
		return x // 10

	def search_mask(self,x):
		return [torch.where(x >= 10, x, NaN)]

class AddOne(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 1;
		self.template = "({} + 1)"
		self.in_arg_types = ["value"]

	def forward(self, x):
		return x + 1

class Append25(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 1;
		self.template = "({}*100 + 25)"
		self.in_arg_types = ["value"]

	def forward(self, x):
		return x*100 + 25


class Equals(BaseOperator):
	def __init__(self):
		super().__init__()

		self.num_flt_inputs = 2;
		self.template = "({} == {})"
		self.commutative = True;
		self.in_arg_types = ["value"]

	def forward(self, x,y):
		eq = x == y
		return torch.where(eq,eq.float(),NaN)


class OperatorGraph(BaseOperator):
	def _gen_expression(self,x,arg_type=None):
		if(isinstance(x,(list,tuple))):
			front = x[0]
			rest = [self._gen_expression(x[j],front.in_arg_types[j-1]) for j in range(1,len(x))]
			return (front,*rest)
		elif(isinstance(x,int)):
			arg = "?foa" + str(len(self.in_args))
			self.in_args.append(arg)
			self.in_arg_types.append(arg_type)
			return arg
		else:
			return x

	def _gen_template(self,x):
		# if(first): self._count = 0
		if(isinstance(x,(list,tuple))):
			rest = [self._gen_template(x[j]) for j in range(1,len(x))]
			return x[0].template.format(*rest)
		elif(isinstance(x,int)):
			# arg = "E" + str(self._count)
			# self._count += 1
			return "{}"
		else:
			return x

	


	def __init__(self,tup,backmap=None,out_arg_types=None):
		super().__init__()
		# self.out_arg_types = out_arg_types
		# if(out_arg_types != None and isinstance(tup,(list,tuple)) and isinstance(tup[0],BaseOperator)):
		# 	assert tup[0].out_arg_types == out_arg_types

		self.in_args = []
		self.in_arg_types = []

		self.template = "g[" + self._gen_template(tup) + "]"
		# print("TEMPLATE",self.template)
		self.expression = self._gen_expression(tup,backmap)

		#TODO: FIX THIS 
		if(self.in_arg_types == [None]):
			self.in_arg_types = ['value']

		self.num_flt_inputs = len(self.in_args)


	def __eq__(self,other):
		return isinstance(other, OperatorGraph) and other.expression == self.expression
	def __hash__(self):
		return hash(self.expression)


		# print(self.expression)
	# def __str__(self):
	# 	return self.template % ('E' + str(i))

	def _forward(self,x,inps,first=True):
		if(first): self._count = 0
		if(isinstance(x,(list,tuple))):
			front = x[0]
			# print(front.forward(*[ self._forward(x[j],inps,False) for j in range(1,len(x))]))
			return front.forward(*[ self._forward(x[j],inps,False) for j in range(1,len(x))])
		else:
			inp = inps[self._count]
			# print(inp.shape)
			self._count += 1
			return inp

	def forward(self,*inps):
		# print([x.shape for x in inps])
		# xd = 
		# print(xd)
		return self._forward(self.expression,inps)

	def compute(self,mapping,state):
		state = state.get_view("flat_ungrounded")
		inps = []
		for arg_type,key in zip(self.in_arg_types, sorted(mapping.keys())):
			# print(self)
			# print(self.in_arg_types)
			# print(self.expression)
			inps.append(torch.tensor(float(state[(arg_type,mapping[key])])))
		# print(inps)
		return self._forward(self.expression,inps).item()	

def reshape_args(args):
	l = len(args)-1
	if(l == -1):
		return None
	return [a.view(*([1]*i + [-1] + (l-i)*[1])) for i, a in enumerate(args)]

def create_redundency_mask(n_prev,n,d):
	# print("n_prev", n_prev,n, n-n_prev)
	# torch.where(torch.ones(n_prev,n_prevm).byte(),torch.zeros(n,n), torch.ones(1))
	return  F.pad(torch.zeros([n_prev]*d),tuple([0,n-n_prev]*d), value=1).byte()
	# return torch.zeros(n,n).byte()


def create_diag_mask(n,d):
	return create_mask(n,d,torch.ne)

def create_upper_triag_mask(n,d):
	return create_mask(n,d,torch.lt)
def create_lower_triag_mask(n,d):
	return create_mask(n,d,torch.gt)

def create_mask(n,d,f=torch.eq):
	a = torch.nonzero(torch.ones([n]*d)).view(-1,1,d).float()
	w = torch.tensor([1,-1]).view(1,1,-1).float()
	sel = (f(torch.conv1d(a,w, None,1,0,1,1), 0)).all(2).view([n]*d)
	return sel.byte()

def resolve_inputs(flat_input_indices, op_inds,n_elems,operator_nf_inps):
	# print(op_inds)

	nf_inps = operator_nf_inps.gather(0,op_inds)
	# print("op_inds:", op_inds)
	# print("nf_inps:",nf_inps)

	# print(nf_inps.shape)
	max_in_degree = torch.max(nf_inps)

	temp_ind = flat_input_indices

	# print("n_elems:", n_elems)
	args = []
	for j in range(max_in_degree-1,-1,-1):
		modulus = n_elems**j
		ok = nf_inps >= j+1
		arg = torch.where(ok,(temp_ind // modulus),torch.tensor(-1))
		args.append(arg.view(-1,1))
		temp_ind = torch.where(ok,temp_ind % modulus,temp_ind)


	# print("args", args)
	# print(max_in_degree)
	# for j in range(max_in_degree):
	# print(flat_input_indices, op_inds)
	# print(nf_inps)
	return torch.cat(args,1)

def resolve_operator(offset_indicies,offset,operator_nf_inps):
	offset_indicies -= offset

	# print("offset_indicies",offset_indicies)

	operator_counts = torch.pow(offset,operator_nf_inps)
	# print("operator_counts", operator_counts)
	cs = torch.cumsum(operator_counts,0) 
	# print("CS",cs)
	# raise ValueError()
	# print(cs.view(-1,1) < offset_indicies.view(1,-1))
	# print(torch.sum(cs.view(-1,1) < offset_indicies.view(1,-1),0))

	op_inds = torch.sum(cs.view(-1,1) <= offset_indicies.view(1,-1),0)
	# print("op_inds", op_inds)
	op_offset = torch.cat([torch.tensor([0]),cs],0).gather(0,op_inds)
	# print("op_offset", op_offset)
	# print("offset_indicies", offset_indicies)
	input_indicies = offset_indicies - op_offset

	# print("-------------")
	# print("input_indicies", input_indicies)
	# print("-------------")

	args = resolve_inputs(input_indicies,op_inds,offset,operator_nf_inps)
	# print(args)

	return op_inds, args
	# for op_ind, arg_set in zip(op_inds, args):
	# 	print(op_ind, arg_set)


	# return op_inds, input_indicies 
	# for input_index, op_ind in zip(input_indicies,op_inds):
	# 	resolve_inputs(input_index.item(), op_ind.item(), )



		# torch.sum(cs < offset_indicies,1)


	# cum_
	# print(op_inds)
	# print(op_offset)

def indicies_to_operator_graph(indicies, d_len,operators,operator_nf_inps):
	# if(len(indicies) != 1):
	# 	print(indicies)
	# print(index > torch.tensor(d_len).view(1,-1))
	# print(torch.tensor(d_len).view(1,-1).shape)
	# d_len = 
	n_lsthn = indicies >= d_len.view(1,-1)
	# offset = torch.sum(n_lsthn.long() * d_len,1)
	# indicies - offset
	# print(n_lsthn)
	# print(offset)
	d_bins = torch.sum(n_lsthn,1)
	# print("d_bins", d_bins)
	# print("d_len", d_len)

	# print(torch.unique(d_bins,return_inverse=True,sorted=True))
	# offset = 0
	for d in range(len(d_len)):
		# print("STARTING D", d, len(d_len))

		offset = d_len[d-1] if d > 0 else 0
		offset_indicies = indicies[(d_bins == d).nonzero()].flatten()#-offset


		# total = d_len[d]-offset

		# print(offset)

		if(d > 0 and offset_indicies.shape[0] > 0):
			
			op_inds, args = resolve_operator(offset_indicies,offset,operator_nf_inps)
			# print(op_inds)
			# print(args)
			# if(len(indicies) != 1):
			# 	print(d)
			# 	print(offset_indicies[-20:])
			# 	print(op_inds[-20:], args[-20:])
			# raise(ValueError())

			for op_ind, arg_set in zip(op_inds, args):
				# print((arg_set >= 0).nonzero().shape)
				# print(arg_set.shape)
				arg_set = arg_set.gather(0,(arg_set >= 0).nonzero().flatten())

				# print( "OUT" ,(operator_set[op_ind], numerical_values.gather(0,arg_set) ) )
				# print(len(indicies))
				# print([operator_set[op_ind], *[x.item() for x in arg_set]] )
				# print(arg_set)
				# print(op_ind)
				
				yield [operators[op_ind], *[x.item() if x.item() < d_len[0] else next(indicies_to_operator_graph(x.view(1),d_len,operators,operator_nf_inps)) for x in arg_set]] 
		else:
			# print("MOOOO")
			for indx in offset_indicies:
				yield indx.item()

		# print(total)
		# print(d, off_depth_d_indicies)
		# print(offset)

		# offset += d_len

import types
def repr_rule(x,numerical_values, include_element=True, include_value=True):
	if(isinstance(x,(list,tuple))):
		# print(x[0].template,tuple([repr_rule(y,numerical_values) for y in x[1:]]))
		return x[0].template.format(*[repr_rule(y,numerical_values) for y in x[1:]])
	elif(isinstance(x,types.GeneratorType)):
		return repr_rule(next(x), numerical_values)
	else:
		a = "E" + str(x) if include_element else ""
		b = ":" if include_element and include_value else ""
		c = repr(numerical_values.gather(0,torch.tensor(x)).item()) if include_value else ""
		return a + b + c

def rule_inputs(x):
	# print(x)
	if(isinstance(x,(list,tuple))):
		if(len(x) == 0): return x
		# print("X", x)
		# print(type(list(itertools.chain.from_iterable([rule_inputs(y) for y in x[1:]]))))
		return list(itertools.chain.from_iterable([rule_inputs(y) for y in x[1:]]))
	elif(isinstance(x,types.GeneratorType)):
		return rule_inputs([y for y in x])
	else:
		return [x]



def state_to_tensors(state):
	numerical_values = []
	backmap = []
	for key, val in state.items():
		# print(key,val)

		if(key[0] == "value" and not isinstance(val,bool) and val != ""):
			# TODO: do this better
			# if(isinstance(val,(int,float)) or (isinstance(val,str) and is_numerical.match(val))):
			try:
				numerical_values.append(float(val))
			except ValueError:
				continue

			# print(key,val)
			backmap.append(key)


	return torch.tensor(numerical_values),backmap


def _broadcasted_apply(o,numerical_values,d_len,reshape_set):
	x_d = o.forward(*o.search_mask(*reshape_set[o.num_flt_inputs]))
	# print(numerical_values.shape[0],o.num_flt_inputs)
	# mask = redundency_mask
	mask = create_redundency_mask(d_len[-2] if len(d_len) >= 2 else 0,numerical_values.shape[0],o.num_flt_inputs)
	if(o.num_flt_inputs > 1):
		if(o.commutative):	
			mask = mask & create_upper_triag_mask(numerical_values.shape[0],o.num_flt_inputs)
		else:
			mask = mask & create_diag_mask(numerical_values.shape[0],o.num_flt_inputs)
	x_d = torch.where(mask, x_d, torch.tensor(float('NaN')))
	return x_d

def forward_one(numerical_values,d_len,operators):
	most_args = max([x.num_flt_inputs for x in operators])
	reshape_set = [reshape_args([numerical_values]*i) for i in range(most_args+1)] 
	forwards = [numerical_values]
	# print(redundency_mask.shape)
	for j,o in enumerate(operators):
		x_d = _broadcasted_apply(o,numerical_values,d_len,reshape_set)
		# print(x_d)
		forwards.append(torch.flatten(x_d))
	numerical_values = torch.cat(forwards)
	d_len.append(numerical_values.shape[0])
	return numerical_values, d_len

def to_rule_expression(tup, backmap):
	if(isinstance(x,(list,tuple))):
		front = x[0]
		rest = [to_rule_expression(x[j]) for j in range(1,len(x))]
		return tuple(front,*rest)
	elif(isinstance(x,int)):
		return backmap[x]
	else:
		return x

def how_search(state,
				goal, search_depth = 1,
				operators= None,
				backmap=None,
				allow_bottomout=True,
				allow_copy=True):
	#TODO allow to work on any input arg type
	try:
		goal = float(goal)
	except ValueError:
		if(allow_bottomout):
			yield goal, {}
		return

	#TODO: Find source of weird memory overflow when -1 is a goal
	if(goal == -1):
		goal = 1
	print("GOAL:", goal)
	
	if(operators == None): operators = self.function_set

	operator_nf_inps = torch.tensor([x.num_flt_inputs for x in operators])

	if(isinstance(state, dict)):
		numerical_values,backmap = state_to_tensors(state)
	else:
		numerical_values = state

	assert backmap != None, "backmap needs to exist"

	with torch.no_grad():
		
		d_len = [numerical_values.shape[0]] 
		exp_exists = False
		if(d_len[0] != 0):
			
			# if(not )
			# print(search_depth)
			for d in range(search_depth):
				numerical_values, d_len = forward_one(numerical_values, d_len,operators)

				# print(numerical_values.shape)
				# print(type(numerical_values))

			indicies = (numerical_values == goal).nonzero()

			# print(numerical_values[indicies[-20:]])
			# print(indicies)
			# print("d_len",d_len)
			# print("NUM RESULTS", indicies.shape)
			
			for tup in indicies_to_operator_graph(indicies,torch.tensor(d_len),operators,operator_nf_inps):
				if(not allow_copy and isinstance(tup, int)):
					continue
				inps = rule_inputs(tup)
				if(len(set(inps)) == len(inps) and inps == sorted(inps)):
					exp_exists = True
					og = OperatorGraph(tup)
					vals = [backmap[x][1] for x in inps]
					yield og, {k:v for k,v in zip(og.in_args,vals)}
		if(allow_bottomout and not exp_exists):
			yield goal,{}


# operator_class_set = [AddOne,Append25,Multiply,Div10]
operator_class_set = [Add,Subtract,Multiply,Divide]
# operator_class_set = [Add,Add3,Mod10,Div10]
function_set = [c() for c in operator_class_set ]

class VectorizedPlanner(BasePlanner):
	def __init__(self,search_depth,**kwargs):

		
		self.function_set = function_set
		self.feature_set = []
		self.epsilon = 0.0

		self.search_depth = search_depth


	def apply_featureset(self, state, operators=None):
		# tup = Tuplizer()
		# flt = Flattener()
		# state = flt.transform(tup.transform(state))
		# state = state.get_view("flat_ungrounded")
		return state
		#TODO: Make work
		'''
		if(operators == None):
			operators = self.feature_set

		if(isinstance(state, dict)):
			numerical_values = state_to_tensor(state)
		else:
			numerical_values = state

		with torch.no_grad():
			d_len = [numerical_values.shape[0]] 
				
			numerical_values, d_len = forward_one(numerical_values, d_len,feature_set)

			indicies = (~torch.isnan(numerical_values)).nonzero()
			# print("d_len",d_len)
			# print("NUM RESULTS", indicies.shape)
			for tup in indicies_to_operator_graph(indicies,torch.tensor(d_len),feature_set):
				inps = rule_inputs(tup)
				if(len(set(inps)) == len(inps) and inps == sorted(inps)):
					# to_rule_expression
					# print(tup)
					print(repr_rule(tup,numerical_values))
	'''



	def how_search(self,state,
					sai,
					operators=None,
					foci_of_attention=None,
					search_depth=None,
					allow_bottomout=True,
					allow_copy=True,
					epsilon=0.0):
		goal = sai.inputs["value"]
		state = state.get_view("flat_ungrounded")

		if(operators == None and sai.action == "ButtonPressed"):
			yield -1,{}
			return

		# print("search_depth",search_depth)
		if(search_depth == None): search_depth = self.search_depth
		if(operators == None): operators = self.function_set

		# print("foci_of_attention:",foci_of_attention)
		# print(state)
		if(foci_of_attention != None):
			state = {k:v for k,v in state.items() if k[1].replace("?ele-","") in foci_of_attention}
		# print(state)

		how_itr = how_search(state,goal,search_depth=search_depth,
							operators=operators,
							allow_bottomout=allow_bottomout,
							allow_copy=allow_copy)
		for expr,mapping in how_itr:
			if(foci_of_attention != None and len(foci_of_attention) != len(mapping)):
				print("continue",expr,mapping)
				continue
			print(expr)
			# print(expr,type(expr))
			# print(mapping,type(mapping))
			# print("From HOW:", search_depth)
			# print(type(expr))
			# print(expr,list(mapping.values()))
			yield expr,mapping

		

	def eval_expression(self,x,mapping,state,function_set=None,epsilon=0.0):
		if(isinstance(x,(list,tuple))):
			front = x[0]
			rest = [self.eval_expression(x[j],mapping,state) for j in range(1,len(x))]
			if(isinstance(front,OperatorGraph)):
				# print(",*rest)",rest)
				return (front.compute(mapping,state),*rest)
			#Not sure the rest of the cases below will work
			elif (isinstance(front,BaseOperator)):
				return front.forward(*rest)
			else:
				return (front,*rest)
		else:
			return x

PLANNERS["vectorized"] = VectorizedPlanner

if __name__ == "__main__":

	# import sys
 #    sys.path.insert(0, './')

	g = OperatorGraph("x")

	# g2 = OperatorGraph(g)
	# print(g.tup)
	print(g.in_args)
	# print(g2.in_args)

	for x in how_search({"A":1, "B":2},"x",operators=[g]):
		print(x)
	# print(g)