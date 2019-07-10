import gym
from gym.envs.toy_text import taxi
from agents.ModularAgent import ModularAgent
from agents.QlearnerAgent import QlearnerAgent
from planners.fo_planner import Operator


taxi_full = Operator(('taxi_full', '?v'), [(('loc', '?passenger'), '?pass_loc')], [(('taxi_full', '?passenger'), (lambda x: x == 4, '?pass_loc'))]) 


south = Operator(('South'), [], [('sai', 'board', 'make_move', 'south'), True])
north = Operator(('North'), [], [('sai', 'board', 'make_move', 'north'), True])
east = Operator(('East'), [], [('sai', 'board', 'make_move', 'west'), True])
west = Operator(('West'), [], [('sai', 'board', 'make_move', 'west'), True])

#move = Operator(('Move'), [], [('sai', 'board', 'move', ), True])

pickup = Operator(('sai', 'board', 'make_move', 'pickup'), [], [('sai', 'board', 'make_move', 'pickup'), True])
dropoff = Operator(('sai', 'board', 'make_move', 'dropoff'), [], [('sai', 'board', 'make_move', 'dropoff'), True])

move = Operator(('sai', 'board', 'make_move', '?v'), 
	[], 
	[('sai', 'board', 'make_move', (('move_type', '?v')), True)]
	)


feature_set = [taxi_full] 
function_set = [move]
action_list = ['south', 'north', 'east', 'west', 'pickup', 'dropoff']
state_params = ['taxi_row', 'taxi_col', 'pass_loc', 'dest_idx']
locations = [(0,0), (0,4), (4,0), (4,3)]



def create_state(observation):
	state = {'taxi': {'row': observation[0], 'col': observation[1]},
			 'passenger': {'loc': observation[2]},
			 'dest_idx': {'val': observation[3]}
	}
	return state

def move_toward(p1, p2, alt=None):
	x1, y1 = p1
	x2, y2 = p2
	if x1 > x2:
		return 1
	elif x2 > x1:
		return 0
	elif y1 > y2:
		return 3
	elif y2 > y1:
		return 2
	else:
		return alt

def next_action(prev_state):
	src = (prev_state['taxi']['row'], prev_state['taxi']['col'])
	dest = None
	if prev_state['passenger']['loc'] == 4:
		dest = prev_state['dest_idx']['val']
		action = move_toward(src, locations[dest], alt=5)
	else:
		dest = prev_state['passenger']['loc']
		action = move_toward(src, locations[dest], alt=4)
	x, y = src
	if x==4 and y==0 and dest != 2:
		action = 1
	elif x==3 and y==0 and dest != 2:
		action = 1
	elif x == 2 and y == 0 and dest != 2 and dest != 0:
		action = 2
	elif x==0 and y==1 and dest == 1:
		action = 0
	elif x==1 and y == 1 and (dest ==1 or dest ==3):
		action = 2
	elif x > 2 and y != 0 and (dest == 2):
		action = 1
	elif x <= 2 and x > 0 and y != 0 and dest == 2:
		action = 3
	elif x > 2 and y <=2 and dest == 4:
		action = 1
	elif x >= 1 and x <= 2 and y <=2 and dest == 3:
		action = 2
	elif x ==0 and y == 2 and (dest == 0 or dest == 2):
		action = 0
	elif x == 1 and y == 2 and (dest == 0 or dest == 2):
		action = 3
	elif x == 4 and y == 3 and dest != 3:
		action = 1
	elif x == 3 and y == 3 and dest != 3:
		action = 1
	elif x == 2 and y == 3 and dest != 3 and dest != 1:
		action = 3 
	elif (x == 4 or x == 3) and y == 2 and (dest == 1 or dest == 3):
		action = 1
	elif (x == 2) and y == 2 and (dest == 1 or dest == 3):
		action = 3
	return action

def run(agent, help=True):

	#agent = ModularAgent([], [])

	env = gym.make('Taxi-v2')
	observation = env.reset()
	observation = list(env.decode(observation))
	state = create_state(observation)
	done = False
	steps = 0
	total_reward = 0
	guesses = 0
	while done == False:
		prev_state = state
		# print('observation:{}'.format(list(env.decode(observation))))
		action = 0
		
		response = agent.request(prev_state)
		# get action
		if response == {}:
			# find best action:
			if help:
				action = next_action(prev_state)
			else:
				action = env.action_space.sample()
			guesses += 1
		else:
			action = action_list.index(response['action'])
		#print('action:{}'.format(action))

		observation, reward, done, info = env.step(action)
		total_reward+=reward
		
		env.render()

		#print('reward:{}'.format(reward))
		
		if action > 3:
			selection = 'passenger'
		else:
			selection = 'taxi'
		'''
		agent.train(prev_state, 'board', action_list[action], \
			{'row': prev_state['taxi']['row'], \
			'col': prev_state['taxi']['col'], \
			'loc': prev_state['passenger']['loc'], \
			'val': prev_state['dest_idx']['val']}, reward, None, None, state)
		'''
		observation = list(env.decode(observation))
		state = create_state(observation)

		inputs = {'move_type': action_list[action]}
		agent.train(prev_state, 'board', 'make_move', inputs, reward, None, None, state)
		
		#agent.train(prev_state,selection, action_list[action], {}, reward, None, None, state)
		steps += 1
	#print("Finished in {} steps".format(steps))
	env.close()
	
	return steps, guesses, total_reward

def main():
	iters = 1
	agent = QlearnerAgent(feature_set, [])
	data = []
	for x in range(iters):
		data.append(run(agent))
		print('.')
	print(data)
	#for key, val in agent.q_learner.Q.items():
		#print(key.views)
		#print(val.Q)
	


if __name__ == '__main__':
	main()