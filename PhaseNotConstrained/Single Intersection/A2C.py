"""Actor-Critic based Learning at a single intersection. Vehicles arrive at the intersection
from four different directions. Phase = 0 implies directions 0 and 1 will be open
while phase = 1 implies directions 2 and 3 will be open.

1. No replay buffer is used.
2. No batch processing. The model learns as the data arrives sequentially.
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from collections import deque
from torch.distributions.categorical import Categorical
import os


# Maps states to a distribution on the action space
class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super(Actor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(self.state_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.action_size)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = F.softmax(self.linear3(output), dim=-1)
        return output


# Maps states to value
class Critic(nn.Module):
    def __init__(self, state_size, hidden_size=32):
        super(Critic, self).__init__()
        self.state_size = state_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(self.state_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, 1)

    def forward(self, state):
        output = F.relu(self.linear1(state))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output


def train(args):
    base = os.path.basename(__file__)
    filename = os.path.splitext(base)[0]

    actor = Actor(4, 2)
    critic = Critic(4)

    optimizer = optim.Adam(list(actor.parameters()) + list(critic.parameters()))
    env = args.env

    MeanQ = torch.tensor([0.])  # Rolling average
    QVec = deque([], maxlen=args.rolling_window)

    for iter_count in range(args.run):
        curr_state = env.states.clone()
        # It is important to clone, otherwise it'll pass a pointer

        # Choose the action with the highest probability
        curr_phase = torch.argmax(actor(curr_state)).unsqueeze(0)

        env.simulate(curr_phase)

        next_state = env.states.clone()
        cost = torch.sum(next_state)/args.scaling

        QVec.append(torch.sum(next_state))

        if iter_count < args.rolling_window:
            MeanQ += (QVec[-1] - MeanQ) / (iter_count + 1)
        else:
            MeanQ += (QVec[-1] - QVec[0]) / args.rolling_window

        advantage = cost + args.gamma*critic(next_state).detach() - critic(curr_state)
        policy = Categorical(actor(curr_state))
        log_prob = policy.log_prob(curr_phase)
        entropy = policy.entropy()  # This is to enforce exploration
        loss = log_prob*advantage.detach() + advantage.pow(2) + args.entropy_coeff*entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_count % args.update_cycle == 0:
            print(filename + f": Iteration:{iter_count + 1} and mean queue length: {MeanQ}")

    if not os.path.exists('Models'):
        os.mkdir('Models')
    torch.save(actor.state_dict(), 'Models/' + filename + 'Actor.pkl')
    torch.save(critic.state_dict(), 'Models/' + filename + 'Critic.pkl')


def evaluate(args):
    actor = Actor(4, 2)

    base = os.path.basename(__file__)
    filename = os.path.splitext(base)[0]

    if not os.path.exists('Models/' + filename + 'Actor.pkl'):
        raise ValueError('Model does not exist.')
    actor.load_state_dict(torch.load('Models/' + filename + 'Actor.pkl'))

    env = args.env
    MeanQ = 0

    for iter_count in range(args.run_eval):
        curr_state = env.states.clone()

        # Choose the action with the highest probability
        curr_phase = torch.argmax(actor(curr_state)).unsqueeze(0)
        env.simulate(curr_phase)
        next_state = env.states.clone()
        MeanQ += (torch.sum(next_state) - MeanQ) / (iter_count + 1)

    print(filename + f": Mean queue length after evaluation is: {MeanQ}")
