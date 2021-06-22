import torch
from Environment import TwoIntersections


class Parameters:
    def __init__(self):
        self.arr_rates = 2.0*torch.ones([6])
        self.dept_rates = 10*torch.ones([8])
        self.env = TwoIntersections(self.arr_rates, self.dept_rates)
        self.run = 10**5
        self.run_eval = 10**4
        self.explore_prob = 0.1
        self.gamma = 0.99
        self.memory_capacity = 10**4
        self.rolling_window = 10**4
        self.clip_grad_value = 3
        self.mini_batch_size = 10
        self.update_cycle = 50
        self.scaling = 1000
        self.hidden_size = 32

        # clip_grad_value should be adapted according to arrival rate,
        # This is important for QMIX.
        # Appropriate cost scaling helps in stabilization and getting better results.
