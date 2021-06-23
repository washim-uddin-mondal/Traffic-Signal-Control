"""Deep Q Learning for Grid intersections. All intersections learn
independently.
"""
import os
import torch
import torch.optim as optim
import random
from NeuralNetworks import DQN, QMIX
from collections import deque
from TrafficNetwork import GridNet


def train(args):
    base = os.path.basename(__file__)
    filename = os.path.splitext(base)[0]

    batch_size = args.batch_size
    IntersectionN = args.length * args.width
    qmix_net_main = QMIX(4 * IntersectionN, intersection_num=IntersectionN)
    qmix_net_target = QMIX(4 * IntersectionN, intersection_num=IntersectionN)
    qmix_net_target.load_state_dict(qmix_net_main.state_dict())

    main_nets = []
    target_nets = []
    optim_par = []

    for _ in range(IntersectionN):
        main_nets.append(DQN(4, 2, hidden_size=args.hidden_size))
        target_nets.append(DQN(4, 2, hidden_size=args.hidden_size))
        target_nets[-1].load_state_dict(main_nets[-1].state_dict())
        optim_par += list(main_nets[-1].parameters())

    optim_par += list(qmix_net_main.parameters())
    optimizer = optim.Adam(optim_par)

    env = GridNet(args)
    replay_memory = deque([], maxlen=args.memory_capacity)
    MeanQ = torch.tensor([0.])
    QVec = deque([], maxlen=args.rolling_window)

    curr_phase = torch.zeros([args.length, args.width], dtype=torch.long)

    for iter_count in range(args.run):
        curr_state = env.Qs.clone()
        # It is important to clone, otherwise it'll pass a pointer

        for index in range(IntersectionN):
            idx = int(index / args.width)
            idy = index % args.width
            if random.uniform(0, 1) < args.explore_prob or iter_count < args.explr_period:
                curr_phase[idx][idy] = torch.randint(0, 2, [1])  # Choose a random phase
            else:
                curr_phase[idx][idy] = torch.argmin(main_nets[index](curr_state[idx, idy, :]))

        env.step(curr_phase)
        next_state = env.Qs.clone()
        cost = torch.sum(next_state)

        QVec.append(cost/IntersectionN)
        # Store cost per intersection. Not used in policy computation.
        # Used for result generation.

        if iter_count < args.rolling_window:
            MeanQ += (QVec[-1] - MeanQ) / (iter_count + 1)
        else:
            MeanQ += (QVec[-1] - QVec[0]) / args.rolling_window

        replay_memory.append([curr_state, curr_phase.clone(), next_state, cost/IntersectionN])
        # It is important to clone, otherwise only a pointer will stored

        if batch_size < len(replay_memory):
            batch_samples = random.sample(replay_memory, batch_size)

            b_curr_states = torch.stack([samples[0] for samples in batch_samples], dim=0)
            b_curr_phases = torch.stack([samples[1] for samples in batch_samples], dim=0)
            b_next_states = torch.stack([samples[2] for samples in batch_samples], dim=0)
            b_costs = torch.stack([samples[3] for samples in batch_samples], dim=0)

            target_q = torch.zeros(batch_size, args.length * args.width)
            estimate_q = torch.zeros(batch_size, args.length * args.width)

            for index in range(IntersectionN):
                idx = int(index / args.width)
                idy = index % args.width
                with torch.no_grad():
                    target_q[:, index] = torch.min(target_nets[index](b_next_states[:, idx, idy, :]), dim=-1)[0] / args.scaling
                estimate_q[:, index] = main_nets[index](b_curr_states[:, idx, idy, :])[range(batch_size), b_curr_phases[:, idx, idy]] / args.scaling

            with torch.no_grad():
                t_out = qmix_net_target(b_next_states.reshape(batch_size, args.length * args.width, -1), target_q)
                target = b_costs + args.gamma * t_out

            estimate = qmix_net_main(b_curr_states.reshape(batch_size, args.length * args.width, -1), estimate_q)
            error = (target - estimate).pow(2)
            loss = torch.mean(error)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(optim_par, args.clip_grad_value)
            optimizer.step()

        if iter_count % args.update_cycle == 0:
            for index in range(IntersectionN):
                target_nets[index].load_state_dict(main_nets[index].state_dict())
            qmix_net_target.load_state_dict(qmix_net_main.state_dict())
            print(filename + f": Iteration:{iter_count + 1} and mean queue length: {MeanQ}")

    if not os.path.exists('Models'):
        os.mkdir('Models')

    index = 0
    for idx in range(args.length):
        for idy in range(args.width):
            torch.save(main_nets[index].state_dict(), f'Models/' + filename + f'{idx}{idy}.pkl')
            index += 1


def evaluate(args):
    base = os.path.basename(__file__)
    filename = os.path.splitext(base)[0]

    main_nets = []

    for idx in range(args.length):
        for idy in range(args.width):
            main_nets.append(DQN(4, 2, hidden_size=args.hidden_size))
            try:
                main_nets[-1].load_state_dict(torch.load(f'Models/' + filename + f'{idx}{idy}.pkl'))
            except ValueError:
                print(f'Model ' + filename + f'{idx}{idy}.pkl does not exist.')

    env = GridNet(args)
    MeanQ = 0

    for iter_count in range(args.run_eval):
        curr_state = env.Qs.clone()
        curr_phase = torch.zeros([args.length, args.width])

        index = 0
        for idx in range(args.length):
            for idy in range(args.width):
                curr_phase[idx][idy] = torch.argmin(main_nets[index](curr_state[idx, idy, :]))
                index += 1

        env.step(curr_phase)
        next_state = env.Qs.clone()
        MeanQ += (torch.sum(next_state)/(args.length * args.width) - MeanQ) / (iter_count + 1)

    print(filename + f": Mean queue after evaluation is: {MeanQ}")
