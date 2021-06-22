import torch
import torch.nn as nn
import torch.nn.functional as F


class RQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super(RQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.layer1 = nn.Linear(self.state_size, self.hidden_size)
        self.layer2 = nn.GRUCell(self.hidden_size, self.hidden_size)
        self.layer3 = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, state, hidden_state):
        output = F.relu(self.layer1(state))
        hidden_state = self.layer2(output, hidden_state)
        output = self.layer3(hidden_state)
        return output, hidden_state


class DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(self.state_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output


class SimpleQMIX(nn.Module):
    def __init__(self, state_size, intersection_num=2, hidden_size=32):
        super(SimpleQMIX, self).__init__()
        self.state_size = state_size
        self.intersection_num = intersection_num
        self.hidden_size = hidden_size
        self.hyper_w = nn.Sequential(nn.Linear(self.state_size, self.hidden_size),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_size, self.intersection_num))
        self.hyper_b = nn.Linear(self.state_size, 1)

    def forward(self, state, q_value):
        q_value = q_value.view(-1, 1, self.intersection_num)
        state = state.reshape(-1, self.state_size)

        w = torch.abs(self.hyper_w(state)).view(-1, self.intersection_num, 1)
        b = self.hyper_b(state).view(-1, 1, 1)

        q_total = torch.bmm(q_value, w) + b
        q_total = q_total.squeeze(-1)

        return q_total


class QMIX(nn.Module):
    def __init__(self, state_size, intersection_num=2, hidden_size=32):
        super(QMIX, self).__init__()
        self.state_size = state_size
        self.intersection_num = intersection_num
        self.hidden_size = hidden_size

        self.hyper_weight0 = nn.Sequential(nn.Linear(self.state_size, self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size, self.intersection_num * self.hidden_size)
                                           )

        self.hyper_weight1 = nn.Sequential(nn.Linear(self.state_size, self.hidden_size),
                                           nn.ReLU(),
                                           nn.Linear(self.hidden_size, self.hidden_size)
                                           )

        self.hyper_bias0 = nn.Linear(self.state_size, self.hidden_size)
        self.hyper_bias1 = nn.Linear(self.state_size, 1)

    def forward(self, state, q_value):
        q_value = q_value.view(-1, 1, self.intersection_num)
        state = state.reshape(-1, self.state_size)

        w0 = torch.abs(self.hyper_weight0(state)).view(-1, self.intersection_num, self.hidden_size)
        b0 = self.hyper_bias0(state).view(-1, 1, self.hidden_size)

        hidden = torch.bmm(q_value, w0) + b0
        hidden = F.elu(hidden)

        w1 = torch.abs(self.hyper_weight1(state)).view(-1, self.hidden_size, 1)
        b1 = self.hyper_bias1(state).view(-1, 1, 1)

        q_total = (torch.bmm(hidden, w1) + b1).view(-1)

        return q_total
