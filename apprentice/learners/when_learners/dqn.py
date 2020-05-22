import torch.nn as nn


class DQN(nn.Module):
    """
    A DQN Architecture that takes separate inputs representing a state-action
    pair returns a single value estimate.
    """

    def __init__(self, n_inputs: int, n_hidden: int = None):
        """
        Specify the number of inputs. Also, specify the number of nodes in each
        hidden layer.  If no value is provided for the number of hidden, then
        it is set to half the number of inputs.
        """
        super(DQN, self).__init__()

        if n_hidden is None:
            n_hidden = (n_inputs + 2) // 2

        self.value = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1)
        )

    def forward(self, x):
        return self.value(x)
