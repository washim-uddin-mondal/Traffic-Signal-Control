import torch
from TrafficNetwork import GridNet


def evaluate(args):
    env = GridNet(args)
    MeanQ = 0
    IntersectionN = args.width * args.length

    curr_state = torch.zeros([IntersectionN, 6])
    next_state = torch.zeros([IntersectionN, 6])
    """
    curr_state[:, :4] = queue lengths
    curr_state[:, 4] = phase
    curr_state[:, 5] = phase time

    Notice that next_state[:, 4] can be interpreted as current action.
    """

    for iter_count in range(args.run_eval):
        curr_state[:, :4] = env.Qs.clone().reshape(-1, 4)
        # It is important to clone, otherwise it'll pass a pointer

        for index in range(IntersectionN):
            # Phase update
            if curr_state[index, 5] < args.min_phase_time:
                next_state[index, 4] = curr_state[index, 4]
            elif curr_state[index, 5] < args.max_phase_time:
                next_state[index, 4] = torch.randint(0, 2, [1])  # Choose a random phase
            else:
                next_state[index, 4] = (1 + curr_state[index, 4]) % 2

            # Phase time update
            if curr_state[index, 4] == next_state[index, 4]:
                next_state[index, 5] = curr_state[index, 5] + 1
            else:
                next_state[index, 5] = 0

        env.step(next_state[:, 4].reshape(-1, args.width))
        next_state[:, :4] = env.Qs.clone().reshape(-1, 4)

        MeanQ += (torch.sum(next_state[:, :4])/IntersectionN - MeanQ) / (iter_count + 1)
        curr_state[:, 4:] = next_state[:, 4:]

    print(f"Baseline: Mean queue after evaluation is: {MeanQ}")
