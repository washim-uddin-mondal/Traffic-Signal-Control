import torch


def evaluate(args):
    env = args.env
    MeanQ = 0

    for iter_count in range(args.run_eval):
        curr_phase0 = torch.randint(0, 2, [1])
        curr_phase1 = torch.randint(0, 2, [1])

        env.simulate(2 * curr_phase0 + curr_phase1)
        next_state = env.states.clone()
        MeanQ += (torch.sum(next_state) - MeanQ) / (iter_count + 1)

    print(f"Average queue after evaluation is: {MeanQ}")

