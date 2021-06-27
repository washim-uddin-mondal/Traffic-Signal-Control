"""
GridNet:
1. A rectangular grid of intersections. It is represented as a directed graph.
2. The state of the network is represented by a LxWx4 matrix.
3. Vehicles can either move straight or take left turn at each intersection.
4. Length = Number of intersections placed in vertical lines.
5. Width = Number of intersections placed in horizontal lines.

Here is an example of node indexing in the network.

(0, 0)    (0, 1)  ...  (0, W-1)
(1, 0)    (1, 1)  ...  (1, W-1)
.           .             .
.           .             .
.           .             .
(L-1, 0) (L-1, 1) ... (L-1, W-1)
"""

import torch
from numpy.random import binomial


class GridNet:
    def __init__(self, args):
        self.length = args.length
        self.width = args.width
        self.arr_rates = args.arr_rates
        self.d_dept_rate = args.d_dept_rate
        self.l_dept_rate = args.l_dept_rate
        self.gen_rate = args.gen_rate
        self.sink_rate = args.sink_rate
        self.turn_ratio = args.turn_ratio

        self.Qs = torch.zeros(self.length, self.width, 4)

        # self.Qs[:, :, 0] = East bound queues
        # self.Qs[:, :, 1] = North bound queues
        # self.Qs[:, :, 2] = West bound queues
        # self.Qs[:, :, 3] = South bound queues

        self.GenRates = self.gen_rate * torch.ones(self.length, self.width, 4)
        self.GenRates[:, 0, 0] = self.arr_rates[0] * torch.ones(self.length)    # Boundary edges
        self.GenRates[-1, :, 1] = self.arr_rates[1] * torch.ones(self.width)
        self.GenRates[:, -1, 2] = self.arr_rates[2] * torch.ones(self.length)
        self.GenRates[0, :, 3] = self.arr_rates[3] * torch.ones(self.width)

        self.SinkRates = self.sink_rate * torch.ones(self.length, self.width, 4)
        self.SinkRates[:, 0, 0] = torch.zeros(self.length)                      # Boundary edges
        self.SinkRates[-1, :, 1] = torch.zeros(self.width)
        self.SinkRates[:, -1, 2] = torch.zeros(self.length)
        self.SinkRates[0, :, 3] = torch.zeros(self.width)

        self.DeptRatesD = self.d_dept_rate * torch.ones(self.length, self.width, 4)
        self.DeptRatesL = self.d_dept_rate * torch.ones(self.length, self.width, 4)

    def step(self, Phases):
        if not self.sink_rate == 0:
            SinkQs = torch.min(self.Qs, torch.poisson(self.SinkRates))
            self.Qs -= SinkQs

        PotentialQsLeftTurn = torch.from_numpy(binomial(self.Qs, p=self.turn_ratio))
        PotentialQsDir = self.Qs - PotentialQsLeftTurn

        DeptQsD = torch.min(PotentialQsDir, torch.poisson(self.DeptRatesD))        # Direct departure
        DeptQsD[:, :, 0] *= (Phases == 0)
        DeptQsD[:, :, 1] *= (Phases == 1)
        DeptQsD[:, :, 2] *= (Phases == 0)
        DeptQsD[:, :, 3] *= (Phases == 1)

        DeptQsL = torch.min(PotentialQsLeftTurn, torch.poisson(self.DeptRatesL))   # Left turn departure
        DeptQsL[:, :, 0] *= (Phases == 0)
        DeptQsL[:, :, 1] *= (Phases == 1)
        DeptQsL[:, :, 2] *= (Phases == 0)
        DeptQsL[:, :, 3] *= (Phases == 1)

        self.Qs -= DeptQsD + DeptQsL
        self.Qs += torch.poisson(self.GenRates)

        if self.width > 1:
            self.Qs[:, 1:, 0] += DeptQsD[:, 0:-1, 0] + DeptQsL[:, 0:-1, 3]
            self.Qs[:, 0:-1, 2] += DeptQsD[:, 1:, 2] + DeptQsL[:, 1:, 1]

        if self.length > 1:
            self.Qs[0:-1, :, 1] += DeptQsD[1:, :, 1] + DeptQsL[1:, :, 0]
            self.Qs[1:, :, 3] += DeptQsD[0:-1, :, 3] + DeptQsL[0:-1, :, 2]
