import json
import torch
import torch.nn as nn
import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader


def toTensorF(x):
    return torch.tensor(x, dtype=torch.float)


def toTensorL(x):
    return torch.tensor(x, dtype=torch.long)


def simpleModel(inFeatures=50, outFeatures=13, hidden=0, tanhOut=False, sigmoidOut=False):
    if hidden == 0:
        if tanhOut:
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(inFeatures, outFeatures),
                nn.Tanh)
        elif sigmoidOut:
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(inFeatures, outFeatures),
                nn.Sigmoid())
        else:
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(inFeatures, outFeatures))
    else:
        if tanhOut:
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(inFeatures, hidden),
                nn.ReLU(),
                nn.Linear(hidden, outFeatures),
                nn.Tanh)
        elif sigmoidOut:
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(inFeatures, hidden),
                nn.ReLU(),
                nn.Linear(hidden, outFeatures),
                nn.Sigmoid())
        else:
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(inFeatures, hidden),
                nn.ReLU(),
                nn.Linear(hidden, outFeatures))


def loadModel(model, filePath, onGPU=False):
    model.load_state_dict(torch.load(
        filePath, map_location=torch.device('cuda' if onGPU else 'cpu')))
    model.eval()
    return model


def saveModel(model, filePath):
    torch.save(model.state_dict(), filePath)


def loadModelV(model, filePath, onGPU=False):
    return loadModel(model, f'D:/Data/Notebook/gamble/torch_model/{filePath}', onGPU)


def saveModelV(model, filePath):
    torch.save(model.state_dict(),
               f'D:/Data/Notebook/gamble/torch_model/{filePath}')


def predict(model, tensorData, threshold=0, defaultValue=None):
    """
    pThreshold : 0..1
    return defaultValue if prob < pThreshold"""
    if threshold == 0:
        return [i.argmax().item() for i in model(tensorData)]
    results = predictWithScore(model, tensorData)
    return [x if p >= threshold else defaultValue for x, p in results]


def predictWithScore(model, tensorData):
    with torch.no_grad():
        a = nn.functional.softmax(model(tensorData), dim=1).max(dim=1)
        return list(zip(a[1].detach().tolist(),
                        a[0].detach().tolist()))


def predictMulitTarget(model, tensorData, multi_positions, probability=False):
    """
    multi_positions=[
        [ob1IndexFrom,ob1IndexEnd],...
    ]
    ex:[[0,6],[6,13]]
    """
    with torch.no_grad():
        result = model(tensorData)
        # n = len(indices)
        results = []
        for iStart, iEnd in multi_positions:
            results.append(nn.Softmax(dim=1)(result[:, iStart:iEnd]))
        data = []
        for i in range(len(result)):
            x = []
            for j in range(len(multi_positions)):
                y = results[j][i]
                if probability:
                    x.append((y.argmax().item(), y.max().item()))
                else:
                    x.append(y.argmax().item())
            data.append(x)
        return data


# def predictMultiObjWProb(model, tensorData, indices):
#     return predictMulitTarget(model, tensorData, indices, True)


class VDataSet(Dataset):
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


def getDataLoader(x, y, batchSize=1024, shuffle=True):
    return DataLoader(VDataSet(x, y), batch_size=batchSize, shuffle=shuffle)


def training(model, trainingData, trainingLabel, lossUse=nn.CrossEntropyLoss(), batchSize=1024, shuffle=True, learningRate=.1, numSteps=10, numStepsPerBatch=10, returnFullLoss=False):
    """
    Args:
    kw is what it'mean.
    learningRate: suport multi lr list-or-tupe, but require numSteps same for each  ex learningRate=(.1,.01,.001), numSteps(100,200,300)


    Return:

    if returnFullLoss is True:
    [[all loss on step 1][step 2]
    else False (default):
    [mean(step1),...]
    """
    dataLoader = getDataLoader(
        trainingData, trainingLabel, batchSize=batchSize, shuffle=shuffle)
    losses = []
    if not isinstance(numSteps, list) and not isinstance(numSteps, tuple):
        numSteps = (numSteps,)
    if not isinstance(learningRate, list) and not isinstance(learningRate, tuple):
        learningRate = (learningRate,)

    for numStep, lr in zip(numSteps, learningRate):
        optim = torch.optim.SGD(model.parameters(), lr=lr)
        iter0 = tqdm.trange(numStep)
        for i in iter0:
            iter1 = iter(dataLoader)
            numBatch = len(iter1)
            losses1 = []
            j = 1
            for x, y in iter1:
                iter0.set_description_str(f'{j:03d}/{numBatch:03d} lr {lr}')
                for _ in range(numStepsPerBatch):
                    optim.zero_grad()
                    out = model(x)
                    loss = lossUse(out, y)
                    loss.backward()
                    optim.step()
                    losses1.append(loss.item())
                j += 1
            losses.append(losses1)
    losses = np.array(losses)
    if not returnFullLoss:
        losses = [np.mean(l) for l in losses]
    return losses


def trainingWithMSELoss(model, trainingData, trainingLabel, learningRate=.1, numSteps=10, batchSize=1024, shuffle=True, numStepsPerBatch=100, showBatchProcess=True, returnFullLoss=False):
    return training(model, trainingData, trainingLabel, lossUse=nn.MSELoss(), learningRate=learningRate, numSteps=numSteps, batchSize=batchSize, shuffle=shuffle, numStepsPerBatch=numStepsPerBatch,  returnFullLoss=returnFullLoss)


def trainingWithCrossEntropyLoss(model, trainingData, trainingLabel, learningRate=.1, numSteps=10, batchSize=1024, shuffle=True, numStepsPerBatch=100, showBatchProcess=True, returnFullLoss=False):
    return training(model, trainingData, trainingLabel, lossUse=nn.CrossEntropyLoss(), learningRate=learningRate, numSteps=numSteps, batchSize=batchSize, shuffle=shuffle, numStepsPerBatch=numStepsPerBatch,  returnFullLoss=returnFullLoss)


def chooseTrainTest(x, y, trainPercent=.9):
    n = len(x)
    ind = np.arange(n)
    np.random.shuffle(ind)
    splitAt = int(n * trainPercent)
    indTrain, indTest = ind[:splitAt], ind[splitAt:]
    return x[indTrain], y[indTrain], x[indTest], y[indTest]


def mean(x):
    """somehow torch don't have mean on boolean"""
    return x.sum()/len(x)


def intToIndicatorVector(x, lengthVector):
    """if int feature work like category you should use it
    3 -> (0,0,0,1,0,0)
    """
    a = np.zeros(lengthVector, int)
    a[x] = 1
    return a


def intToBinaryVector(x, lengthVector):
    """if int feature work like category you should use it
    3 -> (1,1,0,0,0,0)
    """
    b = [int(i) for i in list('{:b}'.format(x))]
    l = lengthVector-len(b)
    l = 0 if l < 0 else l
    c = [0 for _ in range(l)]
    c.extend(b)
    return c


def encodingTimeByHourMinute(hour, minute):
    return (*intToBinaryVector(hour, 5), 0 if minute < 30 else 1)


def encodingTime(t):
    return encodingTimeByHourMinute(t.hour, t.minute)


def _crossEntropyLossMultiTarget(out, label, multi_positions):
    lossUse = nn.CrossEntropyLoss()
    loss = 0
    for k, v in enumerate(multi_positions):
        i, j = v
        loss += lossUse(out[:, i:j], label[:, k])
    return loss / len(multi_positions)


def crossEntropyLossMultiTarget(multi_positions):
    """
    multi_possitions:
    example
    ((0,2),(2,5),(5,9))
    out[:,0:2] for target 1 label[:,0]
    out[:,2:5] for target 2 label[:,1]
    out[:,5:9] for target 3 label[:,2]
    return loss_function
    """
    def lossUse(out, label): return _crossEntropyLossMultiTarget(
        out, label, multi_positions=multi_positions)
    return lossUse


def testingWithCrossEntropyLoss(model, xTest, yTest):
    """
    return {'loss': 0.025, 'probTrue': 1.0, 'avgScore': 0.97}"""
    with torch.no_grad():
        out = model(xTest)
        loss = nn.CrossEntropyLoss()(out, yTest)
        a = nn.functional.softmax(out, dim=1).max(dim=1)
        probRight = mean(a[1] == yTest)
        meanScore = mean(a[0])
        return dict(loss=loss.item(), probTrue=probRight.item(), avgScore=meanScore.item())


def testingWithCrossEntropyLossMultiTarget(model, xTest, yTest, multi_positions):
    """
    return {'loss': 0.025, 'probTrue': 1.0, 'avgScore': 0.97}"""
    with torch.no_grad():
        # xTest, yTest = xtt, ytt
        # multi_positions = [[0, 2], [2, 5], [5, 11]]
        out = model(xTest)
        loss = crossEntropyLossMultiTarget(
            multi_positions=multi_positions)(out, yTest)
        b = []
        c = []
        for k, v in enumerate(multi_positions):
            i, j = v
            a = nn.functional.softmax(out[:, i:j], dim=1).max(dim=1)
            b.append(a[1] == yTest[:, k])
            c.append(a[0])
        probRight = mean(torch.vstack(b).sum(0) == 3)
        meanScore = mean(torch.prod(torch.vstack(c), 0))
        return dict(loss=loss.item(), probTrue=probRight.item(), avgScore=meanScore.item())

# x, y: your data/label


# xTrain, yTrain, xTest, yTest = chooseTrainTest(x, y, trainPercent=.9)
# myModel = simpleModel(inFeatures=20, outFeatures=2, hidden=16)

# losses = trainingWithCrossEntropyLoss(myModel,
#                                       trainingData=xTrain,
#                                       trainingLabel=yTrain,
#                                       learningRate=.1,
#                                       numSteps=50,
#                                       numStepsPerBatch=2,
#                                       batchSize=1024)


# training(model, xtn, ytn,
#          batchSize=48,
#          lossUse=lossUse,
#          learningRate=[1, .1, .01, .001],
#          numSteps=[20, 100, 200, 300],
#          numStepsPerBatch=2)[-1]

def LinearRelu(inF, outF): return nn.Sequential(
    nn.Linear(inF, outF),
    nn.ReLU())


def LinearNormRelu(inF, outF): return nn.Sequential(
    nn.Linear(inF, outF),
    nn.BatchNorm1d(outF),
    nn.ReLU())


class LinearReluResidual(nn.Module):
    """
    outF already count inF: = inF + hidF
    """

    def __init__(self, inF, outF):
        super().__init__()
        self.lrn = LinearRelu(inF, outF-inF)

    def forward(self, x):
        return torch.hstack((self.lrn(x), x))


class LinearNormReluResidual(nn.Module):
    """
    outF already count inF: = inF + hidF
    """

    def __init__(self, inF, outF):
        super().__init__()
        self.lrn = LinearNormRelu(inF, outF-inF)

    def forward(self, x):
        return torch.hstack((self.lrn(x), x))


def get_num_params(model):
    return sum(p.numel() for p in model.parameters())


def list_group_by(arr, by_col_index):
    return [[k for k in arr if k[by_col_index] == j] for j in sorted(set((i[by_col_index] for i in arr)))]


def saveModelAsJson(model, filePath):
    def state_dict_to_v(state_dict, relu_after_norm=True, relu_after_linear=True):

        infer_layer_dict = {'weight_bias': 'linear',
                            'weight_bias_running_mean_running_var_num_batches_tracked': 'norm',
                            'weight_ih_l0_weight_hh_l0_bias_ih_l0_bias_hh_l0_weight_ih_l1_weight_hh_l1_bias_ih_l1_bias_hh_l1': 'rnnx2',
                            'weight_ih_l0_weight_hh_l0_bias_ih_l0_bias_hh_l0': 'rnnx1'}

        def infer_layer(keys):
            i = '_'.join([j for _, j in keys])
            return infer_layer_dict[i]

        v = [i.rsplit('.', maxsplit=1) for i in list(state_dict.keys())]
        z = list_group_by(v, 0)
        a = [[infer_layer(keys), [state_dict[k].detach().numpy().tolist()
                                  for k in ['.'.join(i) for i in keys]]] for keys in z]
        if relu_after_norm:
            a = [[i, ['relu', []]] if i[0] == 'norm' else [i, ] for i in a]
            a = [j for i in a for j in i]
        elif relu_after_linear:
            a = [[i, ['relu', []]] if i[0] == 'linear' else [i, ] for i in a]
            a = [j for i in a for j in i]
        if a[-1][0] == 'relu':
            a = a[:len(a)-1]

        return a
    jsonModel = state_dict_to_v(model.state_dict(),
                                relu_after_norm=True, relu_after_linear=True)
    # return jsonModel

    with open(filePath, 'w') as f:
        # s = {k: v.detach().numpy().tolist()
        #      for k, v in model.state_dict().items()}

        json.dump(jsonModel, f)
    return [i for i, _ in jsonModel]
