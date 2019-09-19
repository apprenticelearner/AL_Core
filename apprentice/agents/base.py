from abc import ABCMeta, abstractmethod

from jsondiff import diff as compute_diff


class BaseAgent(metaclass=ABCMeta):
    prior_state = None

    def request(self, state):
        return self.request_diff(compute_diff(state, self.prior_state))

    @abstractmethod
    def request_diff(self, diff):
        pass

    @abstractmethod
    def train(self, state, selection, action, inputs, reward, skill_label,
              foci_of_attention):
        pass

    @abstractmethod
    def train_stateless(self, selection, action, inputs, reward, skill_label,
                        foci_of_attention):
        pass


if __name__ == "__main__":
    pass
