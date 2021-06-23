import torch
from TrafficNetwork import GridNet


def evaluate(args):
    env = GridNet(args)
    MeanQ = 0

    for iter_count in range(args.run_eval):
        curr_phase = torch.randint(0, 2, [args.length, args.width])

        env.step(curr_phase)
        next_state = env.Qs.clone()
        MeanQ += (torch.sum(next_state)/(args.length * args.width) - MeanQ) / (iter_count + 1)

    print(f"Baseline: Mean queue after evaluation is: {MeanQ}")