import torch
import torch.nn as nn
import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader


def simpleModel(inFeatures=50, outFeatures=13, hidden=16, tanhOut=False, sigmoidOut=False):
"""Work on 90% problems
don't need deeper only wider
"""
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
    if not onGPU:
        model.load_state_dict(torch.load(
            f'{filePath}', map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(
            f'{filePath}', map_location=torch.device('cuda')))
    model.eval()
    return model


def saveModel(model, filePath):
    torch.save(model.state_dict(),
               f'{filePath}')


def predict(model, tensorData, threshold=0, defaultValue=None):
    """
    threshold : 0..1
    return defaultValue if score < threshold"""
    if threshold == 0:
        return [i.argmax().item() for i in model(tensorData)]
    results = predictWithScore(model, tensorData)
    return [x if p >= threshold else defaultValue for x, p in results]


def predictWithScore(model, tensorData):
    with torch.no_grad():
        a = nn.functional.softmax(model(tensorData), dim=1).max(dim=1)
        return list(zip(a[1].detach().tolist(),
                        a[0].detach().tolist()))


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


def training(model, trainingData, trainingLabel, lossUse=nn.CrossEntropyLoss(), batchSize=1024, shuffle=True, learningRate=.1, numSteps=10, numStepsPerBatch=10, showBatchProcess=False, returnFullLoss=False):
    """
    kw is what it'mean.
    showBatchProcess: True: show tqdm process for each batch run. False: only one overrall
    returnFullLoss: True return [[all loss on step 1][step 2]...
    default False: [mean_loss(step1),....]
    """
    dataLoader = getDataLoader(
        trainingData, trainingLabel, batchSize=batchSize, shuffle=shuffle)
    optim = torch.optim.SGD(model.parameters(), lr=learningRate)
    losses = []
    iter0 = np.arange(numSteps)
    if not showBatchProcess:
        iter0 = tqdm.tqdm(iter0)
    for i in iter0:
        iter1 = iter(dataLoader)
        if showBatchProcess:
            iter1 = tqdm.tqdm(iter1, desc=f'{i+1}/{numSteps}')
        losses1 = []
        for x, y in iter1:
            for _ in range(numStepsPerBatch):
                optim.zero_grad()
                out = model(x)
                loss = lossUse(out, y)
                loss.backward()
                optim.step()
                losses1.append(loss.item())
        losses.append(losses1)
    losses = np.array(losses)
    if not returnFullLoss:
        losses = [np.mean(l) for l in losses]
    return losses


def trainingWithMSELoss(model, trainingData, trainingLabel, learningRate=.1, numSteps=10, batchSize=1024, shuffle=True, numStepsPerBatch=100, showBatchProcess=True, returnFullLoss=False):
    return training(model, trainingData, trainingLabel, lossUse=nn.MSELoss(), learningRate=learningRate, numSteps=numSteps, batchSize=batchSize, shuffle=shuffle, numStepsPerBatch=numStepsPerBatch, showBatchProcess=showBatchProcess, returnFullLoss=returnFullLoss)


def trainingWithCrossEntropyLoss(model, trainingData, trainingLabel, learningRate=.1, numSteps=10, batchSize=1024, shuffle=True, numStepsPerBatch=100, showBatchProcess=True, returnFullLoss=False):
    return training(model, trainingData, trainingLabel, lossUse=nn.CrossEntropyLoss(), learningRate=learningRate, numSteps=numSteps, batchSize=batchSize, shuffle=shuffle, numStepsPerBatch=numStepsPerBatch, showBatchProcess=showBatchProcess, returnFullLoss=returnFullLoss)


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
