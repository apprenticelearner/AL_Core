from copy import deepcopy
from typing import Collection

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.feature_extraction import FeatureHasher

from apprentice.learners.WhenLearner import WhenLearner
from apprentice.working_memory.representation import Activation
from apprentice.learners.when_learners.actor_critic import ACValueNet
from apprentice.learners.when_learners.actor_critic import ACActionNet
from apprentice.learners.when_learners.replay_memory import Transition


class ActorCriticLearner(WhenLearner):
    def __init__(self, gamma=0.9, lr=1e-3, state_size=1000, action_size=1000,
                 hidden_size=200):
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "cpu")
        self.gamma = gamma
        self.lr = lr

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        self.state_hasher = FeatureHasher(n_features=self.state_size)
        self.action_hasher = FeatureHasher(n_features=self.action_size)
        self.value_net = ACValueNet(self.state_size, self.hidden_size)
        self.action_net = ACActionNet(self.action_size, self.hidden_size,
                                      self.hidden_size)

        params = (list(self.value_net.parameters()) +
                  list(self.action_net.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

    def gen_state_vector(self, state: dict) -> np.ndarray:
        state = {str(a): state[a] for a in state}
        return self.state_hasher.transform([state]).toarray()

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

        return self.action_hasher.transform(action_dicts).toarray()

    def eval_all(self, state: dict,
                 actions: Collection[Activation]) -> Collection[float]:
        pass

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
            return action_val[0].cpu().item()

    def update(
        self,
        state: dict,
        action: Activation,
        reward: float,
        next_state: dict,
        next_actions: Collection[Activation],
    ) -> None:
        return

        sa = self.generate_vector(state, action)
        if len(next_actions) == 0:
            next_sa = None
        else:
            next_sa = np.stack((self.generate_vector(next_state,
                                                     next_actions[i])
                                for i in range(len(next_actions))))

        # print("REWARD")
        # print(reward)
        # print("NEXT SAs")
        # print(next_sa.shape)
        # print()

        self.replay_memory.push(
            torch.from_numpy(sa).float().to(self.device),
            torch.tensor([reward]).to(self.device),
            torch.from_numpy(next_sa).float().to(self.device))

        self.train()


