import sys
import getopt
import torch
import random
import time
from Parameters import Parameters


if __name__ == '__main__':
    params = Parameters()
    AlgoList = ['Baseline', 'DQN', 'RQN', 'A2C']
    argv = sys.argv[1:]

    if len(argv) == 0:
        print('Main.py -a <algo> -i <iterations>')
        sys.exit()

    try:
        opts, _ = getopt.getopt(argv, "a:i:")
    except getopt.GetoptError:
        print('Main.py -a <algo> -i <iterations>')
        sys.exit()

    algo = ''

    for opt, arg in opts:
        if opt == '-a':
            if arg not in AlgoList:
                raise NotImplementedError('Algo not implemented.')
            algo = arg
        elif opt == '-i':
            try:
                int(arg)
            except ValueError:
                raise ValueError('Number of iterations should be integer.')

            if not int(arg) > 0:
                raise ValueError('Number of iterations should be positive.')
            params.run = int(arg)

    random.seed(10)
    torch.manual_seed(10)
    t0 = time.time()

    if algo == 'A2C':
        import A2C
        A2C.train(params)
        params = Parameters()  # Create a new environment for evaluation
        A2C.evaluate(params)
    elif algo == 'DQN':
        import DQN
        DQN.train(params)
        params = Parameters()
        DQN.evaluate(params)
    elif algo == 'RQN':
        import RQN
        RQN.train(params)
        params = Parameters()
        RQN.evaluate(params)
    elif algo == 'Baseline':
        import Baseline
        Baseline.evaluate(params)
    else:
        raise Exception('Algo not found.')

    t1 = time.time()
    print(f'Elapsed time is: {t1 - t0} sec.')
