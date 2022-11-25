# simpletorch

simple pytorch function, let you train/save/use model for data with one line



x,y : your data/label

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
                                      
                                      
 #plot losses
 #plot(losses)
