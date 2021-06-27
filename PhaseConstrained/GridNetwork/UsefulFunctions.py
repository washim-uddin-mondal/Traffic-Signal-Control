import os
import torch


def SaveModels(models, filename):
    if not os.path.exists('Models'):
        os.mkdir('Models')

    for index in range(len(models)):
        torch.save(models[index].state_dict(), f'Models/' + filename + f'{index}.pkl')


def SaveResults(MeanQVec, filename):
    if not os.path.exists('Results'):
        os.mkdir('Results')
    p0 = 'Results/Count' + filename + '.pt'
    p1 = 'Results/Mean' + filename + '.pt'
    p2 = 'Results/SqMean' + filename + '.pt'

    if os.path.exists(p0) and os.path.exists(p1) and os.path.exists(p2):
        count = torch.load(p0)
        Mean = torch.load(p1)
        SqMean = torch.load(p2)

        count += 1
        Mean += (MeanQVec - Mean)/count
        SqMean += (MeanQVec * MeanQVec - SqMean) / count
    else:
        count = torch.tensor([1.])
        Mean = MeanQVec
        SqMean = MeanQVec * MeanQVec

    torch.save(count, p0)
    torch.save(Mean, p1)
    torch.save(SqMean, p2)
