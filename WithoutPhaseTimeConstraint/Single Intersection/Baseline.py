"""A simple baseline policy for comparison.
"""
import torch


def evaluate(args):
    env = args.env
    eval_queue = 0

    for iter_count in range(args.run_eval):
        curr_phase = torch.randint(0, 2, [1])
        env.simulate(curr_phase)
        next_state = env.states.clone()
        eval_queue += (torch.sum(next_state) - eval_queue) / (iter_count + 1)

    print(f"Baseline: Mean queue length after evaluation is: {eval_queue}")
