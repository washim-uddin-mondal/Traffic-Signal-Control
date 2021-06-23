# Grid Network

A rectangular grid of arbitrary length and width is considered. The following algorithms are implemented.

1. Baseline (Random action)
2. IQL 
3. VDN
4. QMIX

QMIX is notoriously unstable, especially when the network size is large. To stabilize it, we first let it learn from a
completely exploratory behaviour (no exploitation) for a certain period. It lets the algorithm build a stable base. Then
we start epsilon-greedy policy for learning. Also, it is important that we scale the Q-outputs before feeding them into
the Q-mixing network.

# Run experiments:

python3 Main.py -a IQL -w 3 -l 3 -i 100000

python3 Main.py -a VDN -w 3 -l 3 -i 100000

python3 Main.py -a QMIX -w 3 -l 3 -s 1000 -c 10 -i 100000

python3 Main.py -a Baseline -w 3 -l 3
