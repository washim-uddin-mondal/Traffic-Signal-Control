import os
import torch
import matplotlib.pyplot as plt


def PlotResults(prefix, display_interval=100):
    Means = {}
    SqMeans = {}
    algos0 = []
    algos1 = []

    for f in os.listdir('Results'):
        if f.startswith('Mean' + prefix) and f.endswith('.pt'):
            algo = f[f.find('Lambda') + len('Lambda') + 1: f.rfind('.pt')]
            Means[algo] = torch.load('Results/' + f)
            algos0.append(algo)

        if f.startswith('SqMean' + prefix) and f.endswith('.pt'):
            algo = f[f.find('Lambda') + len('Lambda') + 1: f.rfind('.pt')]
            SqMeans[algo] = torch.load('Results/' + f)
            algos1.append(algo)

    if not algos0 == algos1:
        print('Some results are missing.')
        return

    if len(algos0) == 0:
        print('No results to be displayed.')
        return

    plt.figure()
    plt.ylabel("Average Queue per Intersection")
    plt.xlabel(f"Iterations (x {display_interval})")

    for algo in algos0:
        m = Means[algo]
        sm = SqMeans[algo]
        sd = (sm - m**2)**0.5     # Standard deviation
        plt.plot(m, label=algo)
        plt.fill_between(range(len(m)), m-sd, m+sd, alpha=0.3)
    plt.legend()
    plt.savefig('Results/' + prefix)
