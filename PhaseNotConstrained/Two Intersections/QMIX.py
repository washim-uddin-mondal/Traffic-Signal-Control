"""Deep Q Learning for two interlinked intersections. Both intersections learn
via QMIX.
1. QMIX is very unstable.
2. Clipping the gradients does help.
3. Exponential decay of the exploration probability helps.
"""
import torch
import torch.optim as optim
import random
from NeuralNetworks import DQN, QMIX
from collections import deque
import os


def train(args):
    base = os.path.basename(__file__)
    filename = os.path.splitext(base)[0]

    batch_size = args.mini_batch_size

    qmix_net_main = QMIX(8)
    qmix_net_target = QMIX(8)
    qmix_net_target.load_state_dict(qmix_net_main.state_dict())
    main_nets = [DQN(4, 2), DQN(4, 2)]
    target_nets = [DQN(4, 2), DQN(4, 2)]
    target_nets[0].load_state_dict(main_nets[0].state_dict())
    target_nets[1].load_state_dict(main_nets[1].state_dict())

    param = list(main_nets[0].parameters()) + list(main_nets[1].parameters()) + list(qmix_net_main.parameters())
    optimizer = optim.Adam(param)

    env = args.env
    replay_memory = deque([], maxlen=args.memory_capacity)

    MeanQ = torch.tensor([0.])
    QVec = deque([], maxlen=args.rolling_window)

    for iter_count in range(args.run):
        curr_state = env.states.clone().detach()
        # It is important to clone, otherwise it'll pass a pointer

        args.explore_prob = max(args.explore_prob*torch.exp(-0.00005*torch.tensor(iter_count)), 0.05)
        # Exponential decay of exploration probability helps in stabilization.

        if random.uniform(0, 1) < args.explore_prob:
            curr_phase0 = torch.randint(0, 2, [1])
            curr_phase1 = torch.randint(0, 2, [1])
        else:
            curr_phase0 = torch.argmin(main_nets[0](curr_state[:4])).unsqueeze(0)
            curr_phase1 = torch.argmin(main_nets[1](curr_state[4:])).unsqueeze(0)

        env.simulate(2*curr_phase0+curr_phase1)
        next_state = env.states.clone().detach()
        cost = torch.sum(next_state)     # Scale the input to the MIXING network

        QVec.append(torch.sum(next_state))

        if iter_count < args.rolling_window:
            MeanQ += (QVec[-1] - MeanQ) / (iter_count + 1)
        else:
            MeanQ += (QVec[-1] - QVec[0]) / args.rolling_window

        replay_memory.append([curr_state, curr_phase0, curr_phase1, next_state, cost])

        if batch_size <= len(replay_memory):
            batch_samples = random.sample(replay_memory, batch_size)
            batch_curr_states = torch.stack([samples[0] for samples in batch_samples])
            batch_curr_phases0 = torch.stack([samples[1] for samples in batch_samples])
            batch_curr_phases1 = torch.stack([samples[2] for samples in batch_samples])
            batch_next_states = torch.stack([samples[3] for samples in batch_samples])
            batch_costs = torch.stack([samples[4] for samples in batch_samples])

            with torch.no_grad():
                batch_target0 = torch.min(target_nets[0](batch_next_states[:, :4]), dim=-1)[0]/args.scaling
                batch_target1 = torch.min(target_nets[1](batch_next_states[:, 4:]), dim=-1)[0]/args.scaling
                target_q = qmix_net_target(batch_next_states, torch.stack([batch_target0, batch_target1], dim=-1))
                target = batch_costs + args.gamma * target_q

            batch_estimate0 = main_nets[0](batch_curr_states[:, :4])[range(batch_size), batch_curr_phases0[:, 0]]/args.scaling
            batch_estimate1 = main_nets[1](batch_curr_states[:, 4:])[range(batch_size), batch_curr_phases1[:, 0]]/args.scaling
            estimate = qmix_net_main(batch_curr_states, torch.stack([batch_estimate0, batch_estimate1], dim=-1))

            loss = torch.mean((target.detach() - estimate).pow(2))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(param, args.clip_grad_value)
            optimizer.step()

        if iter_count % args.update_cycle == 0:
            target_nets[0].load_state_dict(main_nets[0].state_dict())
            target_nets[1].load_state_dict(main_nets[1].state_dict())
            qmix_net_target.load_state_dict(qmix_net_main.state_dict())
            print(filename + f": Iteration:{iter_count + 1} and training queue length: {MeanQ}")

    if not os.path.exists('Models'):
        os.mkdir('Models')
    torch.save(main_nets[0].state_dict(), 'Models/' + filename + '0.pkl')
    torch.save(main_nets[1].state_dict(), 'Models/' + filename + '1.pkl')


def evaluate(args):
    base = os.path.basename(__file__)
    filename = os.path.splitext(base)[0]

    main_nets = [DQN(4, 2), DQN(4, 2)]

    if not os.path.exists('Models/' + filename + '0.pkl') and os.path.exists('Models/' + filename + '1.pkl'):
        raise ValueError('Model(s) do(es) not exist.')
    main_nets[0].load_state_dict(torch.load('Models/' + filename + '0.pkl'))
    main_nets[1].load_state_dict(torch.load('Models/' + filename + '1.pkl'))

    env = args.env
    MeanQ = 0

    for iter_count in range(args.run_eval):
        curr_state = env.states.clone()
        curr_phase0 = torch.argmin(main_nets[0](curr_state[:4]))
        curr_phase1 = torch.argmin(main_nets[1](curr_state[4:]))

        env.simulate(2*curr_phase0+curr_phase1)
        next_state = env.states.clone()
        MeanQ += (torch.sum(next_state) - MeanQ) / (iter_count + 1)

    print(filename + f": Mean queue after evaluation is: {MeanQ}")
