"""Deep Q Learning for Grid intersections. All intersections learn
via QMIX.
"""
import os
import torch
import torch.optim as optim
import random
from NeuralNetworks import DQN, QMIX
from collections import deque
from TrafficNetwork import GridNet
from UsefulFunctions import SaveModels, SaveResults


def train(args):
    base = os.path.basename(__file__)
    filename = os.path.splitext(base)[0]

    batch_size = args.batch_size
    IntersectionN = args.length * args.width

    qmix_net_main = QMIX(6 * IntersectionN, intersection_num=IntersectionN)
    qmix_net_target = QMIX(6 * IntersectionN, intersection_num=IntersectionN)
    qmix_net_target.load_state_dict(qmix_net_main.state_dict())

    main_nets = []
    target_nets = []
    optim_par = []

    for _ in range(IntersectionN):
        main_nets.append(DQN(6, 2, hidden_size=args.hidden_size))
        target_nets.append(DQN(6, 2, hidden_size=args.hidden_size))
        target_nets[-1].load_state_dict(main_nets[-1].state_dict())
        optim_par += list(main_nets[-1].parameters())

    optim_par += list(qmix_net_main.parameters())
    optimizer = optim.Adam(optim_par)

    env = GridNet(args)
    replay_memory = deque([], maxlen=args.memory_capacity)
    MeanQ = torch.tensor([0.])
    QVec = deque([], maxlen=args.rolling_window)
    MeanQVec = torch.tensor([])

    curr_state = torch.zeros([IntersectionN, 6])
    next_state = torch.zeros([IntersectionN, 6])
    """
    curr_state[:, :4] = queue lengths
    curr_state[:, 4] = phase
    curr_state[:, 5] = phase time
    
    Notice that next_state[:, 4] can be interpreted as current action.
    """

    for iter_count in range(args.run):
        curr_state[:, :4] = env.Qs.clone().reshape(-1, 4)
        # It is important to clone, otherwise it'll pass a pointer

        for index in range(IntersectionN):
            # Phase update
            if curr_state[index, 5] < args.min_phase_time:
                next_state[index, 4] = curr_state[index, 4]
            elif curr_state[index, 5] < args.max_phase_time:
                if random.uniform(0, 1) < args.explore_prob or iter_count < args.explr_period:
                    next_state[index, 4] = torch.randint(0, 2, [1])  # Choose a random phase
                else:
                    next_state[index, 4] = torch.argmin(main_nets[index](curr_state[index, :]))
            else:
                next_state[index, 4] = (1 + curr_state[index, 4]) % 2

            # Phase time update
            if curr_state[index, 4] == next_state[index, 4]:
                next_state[index, 5] = curr_state[index, 5] + 1
            else:
                next_state[index, 5] = 0

        env.step(next_state[:, 4].reshape(-1, args.width))
        next_state[:, :4] = env.Qs.clone().reshape(-1, 4)
        cost = torch.sum(next_state[:, :4])

        QVec.append(torch.sum(cost)/IntersectionN)
        # Store cost per intersection. Not used in policy computation.
        # Used for result generation.

        if iter_count < args.rolling_window:
            MeanQ += (QVec[-1] - MeanQ) / (iter_count + 1)
        else:
            MeanQ += (QVec[-1] - QVec[0]) / args.rolling_window

        if iter_count % args.display_interval == 0:
            MeanQVec = torch.cat((MeanQVec, MeanQ))

        replay_memory.append([curr_state.clone(), next_state.clone(), cost/IntersectionN])
        # It is important to clone, otherwise only a pointer will stored

        curr_state[:, 4:] = next_state[:, 4:]  # State update for next iteration

        if batch_size < len(replay_memory):
            batch_samples = random.sample(replay_memory, batch_size)

            b_curr_states = torch.stack([samples[0] for samples in batch_samples], dim=0)
            b_next_states = torch.stack([samples[1] for samples in batch_samples], dim=0)
            b_curr_actions = b_next_states[:, :, 4]
            b_costs = torch.stack([samples[2] for samples in batch_samples], dim=0)

            target_q = torch.zeros(batch_size, IntersectionN)
            estimate_q = torch.zeros(batch_size, IntersectionN)

            for index in range(IntersectionN):
                with torch.no_grad():
                    t_out = target_nets[index](b_next_states[:, index, :])
                    case0 = b_next_states[:, index, 5] < args.min_phase_time
                    case1 = b_next_states[:, index, 5] < args.max_phase_time
                    t_out0 = case0 * t_out[range(batch_size), b_next_states[:, index, 4].long()]
                    t_out1 = ~case0 * case1 * torch.min(t_out, dim=-1)[0]
                    t_out2 = ~case1 * t_out[range(batch_size), ((1 + b_next_states[:, index, 4]) % 2).long()]

                    target_q[:, index] = t_out0 + t_out1 + t_out2
                estimate_q[:, index] = main_nets[index](b_curr_states[:, index, :])[range(batch_size), b_curr_actions[:, index].long()]

            with torch.no_grad():
                t_out = qmix_net_target(b_next_states, target_q/args.scaling)
                target = b_costs + args.gamma * t_out

            estimate = qmix_net_main(b_curr_states, estimate_q/args.scaling)

            error = (target - estimate).pow(2)
            loss = torch.mean(error)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(optim_par, args.clip_grad_value)
            optimizer.step()

        if iter_count % args.update_cycle == 0:
            for index in range(IntersectionN):
                target_nets[index].load_state_dict(main_nets[index].state_dict())
            print(filename + f": Iteration:{iter_count + 1} and mean queue length: {MeanQ}")
        if iter_count % args.update_cycle_mix == 0:
            qmix_net_target.load_state_dict(qmix_net_main.state_dict())

    SaveModels(main_nets, filename)
    SaveResults(MeanQVec, f'TrainNet{args.length}x{args.width}Lambda{args.arr_rates[0]}' + filename)


def evaluate(args):
    base = os.path.basename(__file__)
    filename = os.path.splitext(base)[0]

    IntersectionN = args.length * args.width
    main_nets = []

    for index in range(IntersectionN):
        main_nets.append(DQN(6, 2, hidden_size=args.hidden_size))
        try:
            main_nets[-1].load_state_dict(torch.load(f'Models/' + filename + f'{index}.pkl'))
        except ValueError:
            print(f'Model ' + filename + f'{index}.pkl does not exist.')

    env = GridNet(args)
    MeanQ = torch.tensor([0.])
    MeanQVec = torch.tensor([])

    curr_state = torch.zeros([IntersectionN, 6])
    next_state = torch.zeros([IntersectionN, 6])

    for iter_count in range(args.run_eval):
        curr_state[:, :4] = env.Qs.clone().reshape(-1, 4)
        # It is important to clone, otherwise it'll pass a pointer

        for index in range(IntersectionN):
            # Phase update
            if curr_state[index, 5] < args.min_phase_time:
                next_state[index, 4] = curr_state[index, 4]
            elif curr_state[index, 5] < args.max_phase_time:
                next_state[index, 4] = torch.argmin(main_nets[index](curr_state[index, :]))
            else:
                next_state[index, 4] = (1 + curr_state[index, 4]) % 2

            # Phase time update
            if curr_state[index, 4] == next_state[index, 4]:
                next_state[index, 5] = curr_state[index, 5] + 1
            else:
                next_state[index, 5] = 0

        env.step(next_state[:, 4].reshape(-1, args.width))
        next_state[:, :4] = env.Qs.clone().reshape(-1, 4)
        MeanQ += (torch.sum(next_state[:, :4]) / IntersectionN - MeanQ) / (iter_count + 1)
        MeanQVec = torch.cat((MeanQVec, MeanQ))
        curr_state[:, 4:] = next_state[:, 4:]

    print(filename + f": Mean queue after evaluation is: {MeanQ}")
    SaveResults(MeanQVec, f'EvalNet{args.length}x{args.width}Lambda{args.arr_rates[0]}' + filename)
