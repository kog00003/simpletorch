# simpletorch

simple pytorch function, let you train/save/load/use model for data with ease


```python
#x,y : your data/label
#x require torch.float dtype. use torch.tensor(data,dtype=torch.float)
#y torch.float dtype if use MSELoss: torch.tensor(label,dtype=torch.float)
#y torch.long dtype if use CrossEntropyLoss: torch.tensor(label,dtype=torch.long)

#split for train/test with 9/1 ratio
xTrain, yTrain, xTest, yTest = chooseTrainTest(x, y, trainPercent=.9)

#create simple model fit on 90% problems
myModel = simpleModel(inFeatures=20, outFeatures=2, hidden=16)

#training
losses = trainingWithCrossEntropyLoss(myModel,
                                      trainingData=xTrain,
                                      trainingLabel=yTrain,
                                      learningRate=.1,
                                      numSteps=50,
                                      numStepsPerBatch=2,
                                      batchSize=1024)
                                      
                                      
 #plotting
 #plot(losses)
 
 #testing
testResult=testingWithCrossEntropyLoss(myModel, xTest, yTest)
#{'loss': 0.025181233882904053, 'probTrue': 1.0, 'avgScore': 0.9756311774253845}
print(testResult)


# save model
saveModel(myModel, 'mymodel.pt')

# load model
loadModel(myModel, 'mymodel.pt')

# use model
predict(myModel, new_x)

```
