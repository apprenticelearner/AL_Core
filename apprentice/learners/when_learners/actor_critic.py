import torch
import torch.nn as nn


class ValueNet(nn.Module):
    """
    The part of the actor critic network that computes the state value.  Also,
    returns the hidden layer before state valuation, for use in action network.
    """

    def __init__(self, n_inputs: int, n_hidden: int = None):
        """
        Specify the number of inputs. Also, specify the number of nodes in each
        hidden layer.  If no value is provided for the number of hidden, then
        it is set to half the number of inputs.
        """
        super(ValueNet, self).__init__()

        if n_hidden is None:
            n_hidden = (n_inputs + 2) // 2

        self.n_hidden = n_hidden

        self.hidden = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.ReLU()
        )

        self.value = nn.Linear(n_hidden, 1)

    def forward(self, x):
        """
        Returns the value of the state and the hidden layer values.
        """
        x = self.hidden(x)
        return self.value(x), x


class ActionNet(nn.Module):
    """
    The part of the actor critic network that computes the action value.
    """

    def __init__(self, n_action_inputs: int, n_value_hidden: int,
                 n_action_hidden: int = None):
        """
        Takes as input the action features and the hidden values from the value
        net.  Returns a value for the action.
        """
        super(ActionNet, self).__init__()

        if n_action_hidden is None:
            n_action_hidden = (n_action_inputs + n_value_hidden + 2) // 2

        self.hidden = nn.Sequential(
            nn.Linear(n_action_inputs + n_value_hidden, n_action_hidden),
            nn.ReLU()
        )

        self.action_value = nn.Linear(n_action_hidden, 1)

    def forward(self, action_x, value_hidden):
        """
        Returns the value of the state and the hidden layer values.
        """
        x = self.hidden(torch.cat((action_x, value_hidden), 1))
        return self.action_value(x)
