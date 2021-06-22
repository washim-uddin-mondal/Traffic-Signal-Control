"""Deep Q Learning for two interlinked intersections. Both intersections learn
jointly.
"""
import torch
import torch.optim as optim
import random
from NeuralNetworks import DQN
from collections import deque
import os


def train(args):
    base = os.path.basename(__file__)
    filename = os.path.splitext(base)[0]

    batch_size = args.mini_batch_size

    main_net = DQN(8, 4)
    target_net = DQN(8, 4)
    target_net.load_state_dict(main_net.state_dict())

    optimizer = optim.Adam(main_net.parameters())

    env = args.env
    replay_memory = deque([], maxlen=args.memory_capacity)
    MeanQ = torch.tensor([0.])
    QVec = deque([], maxlen=args.rolling_window)

    for iter_count in range(args.run):
        curr_state = env.states.clone()
        # It is important to clone, otherwise it'll pass a pointer

        if random.uniform(0, 1) < args.explore_prob:
            curr_phase = torch.randint(0, 4, [1])
        else:
            curr_phase = torch.argmin(main_net(curr_state)).unsqueeze(0)

        env.simulate(curr_phase)
        next_state = env.states.clone()
        cost = torch.sum(next_state)/args.scaling

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

            batch_target = batch_costs + args.gamma*torch.min(target_net(batch_next_states), dim=-1)[0]
            batch_estimate = main_net(batch_curr_states)[range(batch_size), batch_curr_phases[:, 0]]

            loss = torch.mean((batch_target.detach() - batch_estimate).pow(2))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if iter_count % args.update_cycle == 0:
            target_net.load_state_dict(main_net.state_dict())
            print(f"Iteration:{iter_count + 1} and training queue length: {MeanQ}")

    if not os.path.exists('Models'):
        os.mkdir('Models')
    torch.save(main_net.state_dict(), 'Models/' + filename + '.pkl')


def evaluate(args):
    base = os.path.basename(__file__)
    filename = os.path.splitext(base)[0]

    main_net = DQN(8, 4)

    if not os.path.exists('Models/' + filename + '.pkl'):
        raise ValueError('Model does not exist.')
    main_net.load_state_dict(torch.load('Models/' + filename + '.pkl'))

    env = args.env
    MeanQ = 0

    for iter_count in range(args.run_eval):
        curr_state = env.states.clone()
        curr_phase = torch.argmin(main_net(curr_state))

        env.simulate(curr_phase)
        next_state = env.states.clone()
        MeanQ += (torch.sum(next_state) - MeanQ) / (iter_count + 1)

    print(filename + f": Mean queue after evaluation is: {MeanQ}")
