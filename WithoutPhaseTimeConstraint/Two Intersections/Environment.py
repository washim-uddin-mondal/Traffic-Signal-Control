""" Two interlinked intersections. Each intersections has 3 external arrival and
3 external departure directions. Additionally, each intersection has one interlinked
arrival and one interlinked departure direction. Therefore, the joint state has
8 components.

For intersection 0: phase = 0 means directions 0, 1 closed and 2, 3 open while
phase = 1 means the opposite.

For intersection 1: phase = 0 means directions 5, 6 closed and 7, 8 open while
phase = 1 indicates otherwise.

     0        6
  2--|--3  4--|--5
     1        7
"""
import torch


class TwoIntersections:
    def __init__(self, arr_rates, dept_rates):
        self.arr_rates = arr_rates
        self.dept_rates = dept_rates
        self.states = torch.zeros([8])

    def simulate(self, phases):
        if phases == 0:
            phase_vec = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
        elif phases == 1:
            phase_vec = torch.tensor([0, 0, 1, 1, 1, 1, 0, 0])
        elif phases == 2:
            phase_vec = torch.tensor([1, 1, 0, 0, 0, 0, 1, 1])
        elif phases == 3:
            phase_vec = torch.tensor([1, 1, 0, 0, 1, 1, 0, 0])

        curr_dept = torch.poisson(self.dept_rates)
        curr_dept = torch.min(curr_dept, self.states)*phase_vec
        curr_arr = torch.zeros([8])
        curr_arr[:3] = torch.poisson(self.arr_rates[:3])
        curr_arr[3] = curr_dept[5]
        curr_arr[4] = curr_dept[2]
        curr_arr[5:] = torch.poisson(self.arr_rates[3:])

        self.states += curr_arr - curr_dept
