"""Deep Q Learning at a single intersection. Vehicles arrive at the intersection
from four different directions. Phase = 0 implies directions 0 and 1 will be open
while phase = 1 implies directions 2 and 3 will be open.
"""
import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from collections import deque


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


def train(args):
    base = os.path.basename(__file__)
    filename = os.path.splitext(base)[0]

    batch_size = args.mini_batch_size

    main_net = RQN(4, 2, hidden_size=args.hidden_size)
    main_hidden = torch.zeros(batch_size, args.hidden_size)
    target_net = RQN(4, 2, hidden_size=args.hidden_size)
    target_hidden = torch.zeros(batch_size, args.hidden_size)

    target_net.load_state_dict(main_net.state_dict())
    optimizer = optim.Adam(main_net.parameters())

    env = args.env
    replay_memory = deque([], maxlen=args.memory_capacity)
    MeanQ = 0
    QVec = deque([], maxlen=args.rolling_window)

    for iter_count in range(args.run):
        curr_state = env.states.clone().detach()
        # It is important to clone, otherwise it'll pass a pointer

        if random.uniform(0, 1) < args.explore_prob or iter_count < batch_size:
            curr_phase = torch.randint(0, 2, [1])  # Choose a random phase
        else:
            main_outs, main_hidden[0:1, :] = main_net(curr_state.unsqueeze(0), main_hidden[0:1, :])
            main_hidden = main_hidden.detach()
            curr_phase = torch.argmin(main_outs).unsqueeze(0)

        env.simulate(curr_phase)
        next_state = env.states.clone().detach()
        cost = torch.sum(next_state)/args.scaling  # Scaling of cost is necessary.

        QVec.append(torch.sum(next_state))

        if iter_count < args.rolling_window:
            MeanQ += (QVec[-1] - MeanQ) / (iter_count + 1)
        else:
            MeanQ += (QVec[-1] - QVec[0]) / args.rolling_window

        replay_memory.append([curr_state, curr_phase, next_state, cost])

        if batch_size < len(replay_memory):
            batch_samples = random.sample(replay_memory, batch_size)

            batch_curr_states = torch.stack([samples[0] for samples in batch_samples])
            batch_curr_phases = torch.stack([samples[1] for samples in batch_samples])
            batch_next_states = torch.stack([samples[2] for samples in batch_samples])
            batch_costs = torch.stack([samples[3] for samples in batch_samples])

            target_out, target_hidden = target_net(batch_next_states, target_hidden)
            batch_target = batch_costs + args.gamma*torch.min(target_out, dim=-1)[0]

            main_out, main_hidden = main_net(batch_curr_states, main_hidden)
            main_hidden = main_hidden.detach()
            # If not detached, may create problem in gradient computation.

            batch_estimate = main_out[range(batch_size), batch_curr_phases[:, 0]]
            error = (batch_target.detach() - batch_estimate).pow(2)
            loss = torch.mean(error)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if iter_count % args.update_cycle == 0:
            target_net.load_state_dict(main_net.state_dict())
            target_hidden = main_hidden.clone().detach()
            print(filename + f": Iteration:{iter_count + 1} and mean queue length: {MeanQ}")

    if not os.path.exists('Models'):
        os.mkdir('Models')
    torch.save(main_net.state_dict(), 'Models/' + filename + '.pkl')
    torch.save(main_hidden, 'Models/' + filename + 'HiddenS.pkl')


def evaluate(args):
    base = os.path.basename(__file__)
    filename = os.path.splitext(base)[0]

    env = args.env
    main_net = RQN(4, 2, args.hidden_size)

    if not os.path.exists('Models/' + filename + '.pkl') and os.path.exists('Models/' + filename + 'HiddenS.pkl'):
        raise ValueError('Model does not exist.')
    main_net.load_state_dict(torch.load('Models/' + filename + '.pkl'))
    main_hidden = torch.load('Models/' + filename + 'HiddenS.pkl')

    MeanQ = 0

    for iter_count in range(args.run_eval):
        curr_state = env.states.clone()

        main_out, main_hidden[0:1, :] = main_net(curr_state.unsqueeze(0), main_hidden[0:1, :])
        curr_phase = torch.argmin(main_out).unsqueeze(0)

        env.simulate(curr_phase)
        next_state = env.states.clone()
        MeanQ += (torch.sum(next_state) - MeanQ) / (iter_count + 1)

    print(filename + f": Mean queue after evaluation is: {MeanQ}")
