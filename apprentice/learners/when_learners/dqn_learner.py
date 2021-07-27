from typing import Collection

import numpy as np
import torch
import torch.nn.functional as F

from apprentice.learners.WhenLearner import WhenLearner
from apprentice.working_memory.representation import Activation
from apprentice.learners.when_learners.actor_critic import ValueNet
from apprentice.learners.when_learners.actor_critic import ActionNet
from apprentice.learners.when_learners.replay_memory import ReplayMemory
from apprentice.learners.when_learners.replay_memory import Transition
from apprentice.learners.utils import OnlineDictVectorizer

# from concept_formation.trestle import TrestleTree
# from sklearn.feature_extraction import FeatureHasher

import logging
log = logging.getLogger(__name__)


class DQNLearner(WhenLearner):
    def __init__(self, gamma=0.7, lr=3e-5, batch_size=64, mem_capacity=10000,
                 # state_size=394, action_size=257, state_hidden_size=197,
                 state_size=300, action_size=100, state_hidden_size=30,
                 action_hidden_size=122):

        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "cpu")
        # self.device = "cpu" #TODO: make cuda not break elsewhere
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size

        self.state_size = state_size
        self.action_size = action_size
        self.state_hidden_size = state_hidden_size
        self.action_hidden_size = action_hidden_size

        self.state_hasher = OnlineDictVectorizer(n_features=self.state_size)
        self.action_hasher = OnlineDictVectorizer(n_features=self.action_size)

        # special case to make things run faster and drop values
        # self.state_hasher = FractionsStateHasher()
        # self.action_hasher = FractionsActionHasher()

        self.value_net = ValueNet(
            self.state_size, self.state_hidden_size).to(self.device)
        self.action_net = ActionNet(self.action_size, self.state_hidden_size,
                                    self.action_hidden_size).to(self.device)

        # create separate target net for computing future value
        self.target_value_net = ValueNet(self.state_size,
                                         self.state_hidden_size)
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.target_value_net.eval()
        self.target_action_net = ActionNet(self.action_size,
                                           self.state_hidden_size,
                                           self.action_hidden_size)
        self.target_action_net.load_state_dict(self.action_net.state_dict())
        self.target_action_net.eval()

        self.replay_memory = ReplayMemory(mem_capacity)

        params = (list(self.value_net.parameters()) +
                  list(self.action_net.parameters()))
        self.optimizer = torch.optim.AdamW(params, lr=self.lr)

    def update_target_net(self):
        self.target_value_net.load_state_dict(self.value_net.state_dict())
        self.target_action_net.load_state_dict(self.action_net.state_dict())

    def gen_state_vector(self, state: dict) -> np.ndarray:
        state = {str(a): state[a] for a in state}

        return self.state_hasher.transform([state])

    def gen_action_vectors(
            self, actions: Collection[Activation]) -> np.ndarray:

        action_dicts = []
        for action in actions:
            act_d = {}
            name = action.get_rule_name()
            act_d['rulename'] = name
            bindings = action.get_rule_bindings()
            for a, v in bindings.items():
                if isinstance(v, bool):
                    act_d[str(a)] = str(v)
                else:
                    act_d[str(a)] = v
            action_dicts.append(act_d)

        return self.action_hasher.transform(action_dicts)

    def eval(self, state: dict, action: Activation) -> float:
        if state is None:
            return 0

        state_x = torch.from_numpy(
            self.gen_state_vector(state)).float().to(self.device)
        action_x = torch.from_numpy(
            self.gen_action_vectors([action])).float().to(self.device)

        with torch.no_grad():
            state_val, state_hidden = self.value_net(state_x)
            action_val = self.action_net(action_x, state_hidden)
            return state_val[0].cpu().item() + action_val[0].cpu().item()

    def eval_multiple(self, state: dict,
                      actions: Collection[Activation]) -> Collection[float]:
        if state is None:
            return 0

        state_x = torch.from_numpy(
            self.gen_state_vector(state)).float().to(self.device)
        action_x = torch.from_numpy(
            self.gen_action_vectors(actions)).float().to(self.device)

        with torch.no_grad():
            state_val, state_hidden = self.value_net(state_x)
            action_val = self.action_net(action_x,
                                         state_hidden.expand(len(actions), -1))
            return (state_val.expand(len(actions), -1) +
                    action_val).squeeze(1).cpu().tolist()

    def update(
        self,
        state: dict,
        action: Activation,
        reward: float,
        next_state: dict,
        next_actions: Collection[Activation],
    ) -> None:

        state_v = self.gen_state_vector(state)
        action_v = self.gen_action_vectors([action])

        if next_state is None or len(next_actions) == 0:
            next_state_v = None
            next_action_vs = None
        else:
            next_state_v = self.gen_state_vector(next_state)
            next_action_vs = self.gen_action_vectors(next_actions)

        self.replay_memory.push(
            torch.from_numpy(state_v).float().to(self.device),
            torch.from_numpy(action_v).float().to(self.device),
            torch.tensor([reward]).float().to(self.device),
            None if next_state_v is None else
            torch.from_numpy(next_state_v).float().to(self.device),
            None if next_action_vs is None else
            torch.from_numpy(next_action_vs).float().to(self.device)
        )

        self.train()

    def train(self):
        # epochs = (len(replay_memory) // target_update // 2) + 1
        batch_size = self.batch_size
        if len(self.replay_memory) < batch_size:
            batch_size = len(self.replay_memory)
        updates = len(self.replay_memory) // batch_size
        if updates < 20:
            updates = 20
        updates *= 3
        if updates > 200:
            updates = 200

        log.debug('len replay mem =' + str(len(self.replay_memory)))
        loss = []
        for i in range(updates):
            if i % 5:
                self.update_target_net()
            loss.append(self.optimize_model())

    def optimize_model(self):
        batch_size = self.batch_size

        if len(self.replay_memory) < self.batch_size:
            batch_size = len(self.replay_memory)

        transitions = self.replay_memory.sample(batch_size)
        batch = Transition(*zip(*transitions))  # transpose batch

        # Get states, actions, and rewards
        state = torch.cat(batch.state).view(batch_size, self.state_size)
        action = torch.cat(batch.action).view(batch_size, self.action_size)
        reward = torch.stack(batch.reward).view(1, batch_size)

        state_value, state_hidden = self.value_net(state)
        action_value = self.action_net(action, state_hidden)
        state_action_values = state_value + action_value

        # compute mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(
            tuple(map(lambda sa: sa is not None, batch.next_state)),
            device=self.device, dtype=torch.bool)
        any_non_final = non_final_mask.sum() > 0

        if any_non_final:
            non_final_next_state = torch.cat(
                [s for s in batch.next_state
                    if s is not None]).view(-1, self.state_size)
            non_final_next_actions = torch.cat(
                [s for s in batch.next_actions
                    if s is not None]).view(-1, self.action_size)

        # how many actions are available for each state
        next_action_lens = [nas.shape[0] for nas in batch.next_actions
                            if nas is not None]
        next_action_start = [sum(next_action_lens[0:i]) for i in
                             range(len(next_action_lens))]

        # Compute next state action indices from policy net
        if any_non_final:

            with torch.no_grad():
                next_value = self.target_value_net(non_final_next_state)
                non_final_next_state_value, non_final_next_hidden = next_value

                next_state_value_expanded = torch.cat([
                    non_final_next_state_value[i].expand(
                        next_action_lens[i], -1)
                    for i in range(len(next_action_start))], 0)

                next_state_hidden_expanded = torch.cat([
                    non_final_next_hidden[i].expand(next_action_lens[i], -1)
                    for i in range(len(next_action_start))], 0)

                non_final_next_action_value = (
                    next_state_value_expanded + self.target_action_net(
                        non_final_next_actions, next_state_hidden_expanded))

        # Compute value of next state actions from target net
        # Detach, so we don't track gradients, target net not getting updated.
        next_state_values = torch.zeros(batch_size, device=self.device)
        if any_non_final:
            next_state_values[non_final_mask] = torch.tensor([
                non_final_next_action_value.narrow(
                    0, next_action_start[i], next_action_lens[i]).max(0)[0]
                for i in range(len(next_action_start))], device=self.device)

        # next_state_values[non_final_mask] = self.net(
        # non_final_next_sas).gather(
        #         1, non_final_next_sa_idx).detach().squeeze()

        # Calculate the expected state-action value
        with torch.no_grad():
            expected_state_action_values = (
                reward + self.gamma * next_state_values).view(batch_size, 1)

        # print(torch.cat([state_action_values, expected_state_action_values],
        # 1))
        # print(expected_state_action_values)

        self.optimizer.zero_grad()

        loss = F.smooth_l1_loss(state_action_values,
                                expected_state_action_values)

        # perform backprop
        loss.backward()

        # for param in self.value_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        # for param in self.action_net.parameters():
        #     param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss.detach().item()
