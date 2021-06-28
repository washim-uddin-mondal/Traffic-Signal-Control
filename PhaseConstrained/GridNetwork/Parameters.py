"""Contains all the traffic network parameters and NN parameters.
"""


class Parameters:
    def __init__(self):
        self.length = 3
        self.width = 3
        self.gen_rate = 0                    # Traffic generation rate at internal edges
        self.arr_rates = [3, 3, 3, 3]        # External arrival rates at boundary edges for four different directions
        self.d_dept_rate = 10      # Departure rate for vehicles crossing an intersection without turn
        self.l_dept_rate = 10      # Departure rate for vehicles crossing an intersection turning left
        self.turn_ratio = 0.2      # Fraction of vehicles turning left
        self.sink_rate = 0         # Traffic sink rate at internal edges (vehicle reaches its destination within network)
        self.min_phase_time = 10
        self.max_phase_time = 60
        self.display_interval = 100

        self.run = 10**5
        self.run_eval = 10**5
        self.run_experiment = 3
        self.explore_prob = 0.05
        self.explr_period = 10**4
        self.gamma = 0.99
        self.memory_capacity = 10**5
        self.rolling_window = 10**4
        self.clip_grad_value = 5
        self.batch_size = 10
        self.update_cycle = 50
        self.update_cycle_mix = 50
        self.scaling = 100
        self.hidden_size = 32

        # The idea of exploratory period is used for a stable initialization.
        # It is primarily required in QMIX.
