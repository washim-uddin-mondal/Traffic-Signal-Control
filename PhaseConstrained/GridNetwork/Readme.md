# Grid Network

A rectangular grid is considered.

# Run experiments

Options:\
-a: algo\
-i: training iterations in each run\
-s: reward scaling\
-l: length of the network\
-w: width of the network\
-r: number of experiment runs

Default values of these options and of some other variables are defined in Parameter.py

python3 Main.py -a IQL -i 100000 -l 3 -w 3 -r 20\
python3 Main.py -a VDN -i 100000 -l 3 -w 3 -r 20\
python3 Main.py -a QMIX -i 100000 -s 1000 -l 3 -w 3 -r 20

# Caution

Before running a fresh set of experiments, make sure Results folder (if any) is empty. Otherwise results might be corrupted.

# Display results

python3 Main.py -d TrainNet3x3\
python3 Main.py -d EvalNet3x3
