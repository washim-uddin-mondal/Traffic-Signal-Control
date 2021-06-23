import sys
import getopt
import torch
import random
import time
from Parameters import Parameters


if __name__ == '__main__':
    params = Parameters()
    AlgoList = ['Baseline', 'IQL', 'VDN', 'QMIX']
    argv = sys.argv[1:]

    if len(argv) == 0:
        print('Main.py -a <algo> -i <iterations>')
        sys.exit()

    try:
        opts, _ = getopt.getopt(argv, "a:i:c:l:w:s:")
    except getopt.GetoptError:
        print('Main.py -a <algo> -i <iterations> -c <clip> -s <scale> -l <length> -w <width>')
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
        elif opt == '-l':
            try:
                int(arg)
            except ValueError:
                raise ValueError('Network length should be integer.')

            if not int(arg) > 0:
                raise ValueError('Network length should be positive.')
            params.length = int(arg)
        elif opt == '-w':
            try:
                int(arg)
            except ValueError:
                raise ValueError('Network width should be integer.')

            if not int(arg) > 0:
                raise ValueError('Network width should be positive.')
            params.width = int(arg)
        elif opt == '-c':
            try:
                float(arg)
            except ValueError:
                raise ValueError('GradClip should be a float.')

            if not float(arg) > 0:
                raise ValueError('GradientClip should be positive.')
            params.clip_grad_value = arg
        elif opt == '-s':
            try:
                float(arg)
            except ValueError:
                raise ValueError('Scaling should be a float.')

            if not float(arg) > 0:
                raise ValueError('Scaling should be positive.')
            params.scaling = float(arg)

    random.seed(10)
    torch.manual_seed(10)
    t0 = time.time()

    if algo == 'IQL':
        import IQL
        IQL.train(params)
        IQL.evaluate(params)
    elif algo == 'VDN':
        import VDN
        VDN.train(params)
        VDN.evaluate(params)
    elif algo == 'QMIX':
        import QMIX
        QMIX.train(params)
        QMIX.evaluate(params)
    elif algo == 'Baseline':
        import Baseline
        Baseline.evaluate(params)
    else:
        raise Exception('Algo not found.')

    t1 = time.time()
    print(f'Elapsed time is: {t1 - t0} sec.')
