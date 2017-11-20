import csv
import random
import knn
import kmeans
import neural_network
import decision_tree
import pandas as pd
import numpy as np

def loadDataset(filename, split, replaceChars, trainingSet=[] , testSet=[]):
    if(replaceChars):
        df = pd.read_csv(filename, header=None)
        cleanup_nums = {
                0: {"x": 0, "o": 1, "b": 2},
                1: {"x": 0, "o": 1, "b": 2},
                2: {"x": 0, "o": 1, "b": 2},
                3: {"x": 0, "o": 1, "b": 2},
                4: {"x": 0, "o": 1, "b": 2},
                5: {"x": 0, "o": 1, "b": 2},
                6: {"x": 0, "o": 1, "b": 2},
                7: {"x": 0, "o": 1, "b": 2},
                8: {"x": 0, "o": 1, "b": 2},
        }
        df.replace(cleanup_nums, inplace=True)
        dataset = df.values.tolist()
        for x in range(len(dataset)-1):
            for y in range(len(dataset[x])-1):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])
    else:
        with open(filename, 'r') as csvfile:
            lines = csv.reader(csvfile)
            dataset = list(lines)
            #print(type(dataset))
            for x in range(len(dataset)-1):
                for y in range(len(dataset[x])-1):
                    dataset[x][y] = float(dataset[x][y])
                if random.random() < split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])


def loadDatasetTree(filename, replaceChars):
    training = pd.DataFrame()
    test = pd.DataFrame()
    df = pd.read_csv(filename, header=None)
    if(replaceChars):
        cleanup_nums = {
            0: {"x": 0, "o": 1, "b": 2},
            1: {"x": 0, "o": 1, "b": 2},
            2: {"x": 0, "o": 1, "b": 2},
            3: {"x": 0, "o": 1, "b": 2},
            4: {"x": 0, "o": 1, "b": 2},
            5: {"x": 0, "o": 1, "b": 2},
            6: {"x": 0, "o": 1, "b": 2},
            7: {"x": 0, "o": 1, "b": 2},
            8: {"x": 0, "o": 1, "b": 2},
        }
        df.replace(cleanup_nums, inplace=True)
    split = 0.66

    for row in df.iterrows():
        if random.random() < split:
            training = training.append(row[1])
        else:
            test = test.append(row[1])

    index_last_column = df.columns[-1]
    last_column = training[training.columns[-1:][0]]
    x = []
    y = np.array(last_column)
    for column in df:
        if column != index_last_column:
            x.append(training[column])

    return [x, y, df, test, training]

print ("\n\n---- KMeans - Libras database ----\n")
dataset = []
with open('databases/movement_libras_10.data', 'r') as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)
    for x in range(len(dataset)-1):
        for y in range(len(dataset[x])):
            dataset[x][y] = float(dataset[x][y])
#KMeans with k = 1
k = 1
kmeans.calculateCentroids(dataset, k)
print ("Nº of samples: "  + repr(len(dataset)))
#KMeans with k = 3
k = 3
kmeans.calculateCentroids(dataset, k)
print ("Nº of samples: "  + repr(len(dataset)))
#KMeans with k = 5
k = 5
kmeans.calculateCentroids(dataset, k)
print ("Nº of samples: "  + repr(len(dataset)))
#KMeans with k = 7
k = 7
kmeans.calculateCentroids(dataset, k)
print ("Nº of samples: "  + repr(len(dataset)))


#KNN with database 1
print ("\n\n---- KNN - Tic-Tac-Toe database ----\n")
trainingSet=[]
testSet=[]
predictions=[]
loadDataset('databases/tic-tac-toe.data', 0.66, 1, trainingSet, testSet)
length = len(trainingSet[0]) - 1
#KNN with k = 1
k = 1
for x in range(len(testSet)):
    neighbors = knn.getNeighbors(trainingSet, testSet[x], k, length)
    result = knn.getResponse(neighbors)
    predictions.append(result)
    #print('*** KNN Prediction =' + repr(result) + ', Real Value=' + repr(testSet[x][-1]))
accuracy = knn.calculateAcc(testSet, predictions)

print ("Accuracy: " + repr(accuracy))
print ("K: " + str(k))
print ("Train: " + repr(len(trainingSet)))
print ("Test: "  + repr(len(testSet)))

#KNN with k = 3
trainingSet=[]
testSet=[]
predictions=[]
loadDataset('databases/tic-tac-toe.data', 0.66, 1, trainingSet, testSet)
k = 3
for x in range(len(testSet)):
    neighbors = knn.getNeighbors(trainingSet, testSet[x], k, length)
    result = knn.getResponse(neighbors)
    predictions.append(result)
    #print('*** KNN Prediction =' + repr(result) + ', Real Value=' + repr(testSet[x][-1]))
accuracy = knn.calculateAcc(testSet, predictions)

print ("Accuracy: " + repr(accuracy))
print ("K: " + str(k))
print ("Train: " + repr(len(trainingSet)))
print ("Test: "  + repr(len(testSet)))

#KNN with k = 5
trainingSet=[]
testSet=[]
predictions=[]
loadDataset('databases/tic-tac-toe.data', 0.66, 1, trainingSet, testSet)
k = 5
for x in range(len(testSet)):
    neighbors = knn.getNeighbors(trainingSet, testSet[x], k, length)
    result = knn.getResponse(neighbors)
    predictions.append(result)
    #print('*** KNN Prediction =' + repr(result) + ', Real Value=' + repr(testSet[x][-1]))
accuracy = knn.calculateAcc(testSet, predictions)

print ("Accuracy: " + repr(accuracy))
print ("K: " + str(k))
print ("Train: " + repr(len(trainingSet)))
print ("Test: "  + repr(len(testSet)))

#KNN with k = 7
trainingSet=[]
testSet=[]
predictions=[]
loadDataset('databases/tic-tac-toe.data', 0.66, 1, trainingSet, testSet)
k = 7
for x in range(len(testSet)):
    neighbors = knn.getNeighbors(trainingSet, testSet[x], k, length)
    result = knn.getResponse(neighbors)
    predictions.append(result)
    #print('*** KNN Prediction =' + repr(result) + ', Real Value=' + repr(testSet[x][-1]))
accuracy = knn.calculateAcc(testSet, predictions)

print ("Accuracy: " + repr(accuracy))
print ("K: " + str(k))
print ("Train: " + repr(len(trainingSet)))
print ("Test: "  + repr(len(testSet)))

#KNN with database 2
print ("\n\n---- KNN - Soybean database ----\n")
trainingSet=[]
testSet=[]
predictions=[]
loadDataset('databases/soybean.data', 0.66, 0, trainingSet, testSet)
length = len(trainingSet[0]) - 1
#KNN with k = 1
k = 1
for x in range(len(testSet)):
    neighbors = knn.getNeighbors(trainingSet, testSet[x], k, length)
    result = knn.getResponse(neighbors)
    predictions.append(result)
    #print('*** KNN Prediction =' + repr(result) + ', Real Value=' + repr(testSet[x][-1]))
accuracy = knn.calculateAcc(testSet, predictions)

print ("Accuracy: " + repr(accuracy))
print ("Train: " + repr(len(trainingSet)))
print ("Test: "  + repr(len(testSet)))

#KNN with k = 3
trainingSet=[]
testSet=[]
predictions=[]
loadDataset('databases/soybean.data', 0.66, 0, trainingSet, testSet)
k = 3
for x in range(len(testSet)):
    neighbors = knn.getNeighbors(trainingSet, testSet[x], k, length)
    result = knn.getResponse(neighbors)
    predictions.append(result)
    #print('*** KNN Prediction =' + repr(result) + ', Real Value=' + repr(testSet[x][-1]))
accuracy = knn.calculateAcc(testSet, predictions)

print ("Accuracy: " + repr(accuracy))
print ("Train: " + repr(len(trainingSet)))
print ("Test: "  + repr(len(testSet)))

#KNN with k = 5
trainingSet=[]
testSet=[]
predictions=[]
loadDataset('databases/soybean.data', 0.66, 0, trainingSet, testSet)
k = 5
for x in range(len(testSet)):
    neighbors = knn.getNeighbors(trainingSet, testSet[x], k, length)
    result = knn.getResponse(neighbors)
    predictions.append(result)
    #print('*** KNN Prediction =' + repr(result) + ', Real Value=' + repr(testSet[x][-1]))
accuracy = knn.calculateAcc(testSet, predictions)

print ("Accuracy: " + repr(accuracy))
print ("Train: " + repr(len(trainingSet)))
print ("Test: "  + repr(len(testSet)))

#KNN with k = 7
trainingSet=[]
testSet=[]
predictions=[]
loadDataset('databases/soybean.data', 0.66, 0, trainingSet, testSet)
k = 7
for x in range(len(testSet)):
    neighbors = knn.getNeighbors(trainingSet, testSet[x], k, length)
    result = knn.getResponse(neighbors)
    predictions.append(result)
    #print('*** KNN Prediction =' + repr(result) + ', Real Value=' + repr(testSet[x][-1]))
accuracy = knn.calculateAcc(testSet, predictions)

print ("Accuracy: " + repr(accuracy))
print ("Train: " + repr(len(trainingSet)))
print ("Test: "  + repr(len(testSet)))

#Neural Network
print ("\n\n---- Neural Network ----\n")
neural_network.executeNetworkTraining()

#Decision Tree
print("\n\n---- Decision Tree - Tic-Tac-Toe ----\n")

#Reading file and setting up the correct database format
data = loadDatasetTree('databases/tic-tac-toe.data', 1)

#Build the decision tree and testing it
decisionTreeObj = decision_tree.buildTree(data[0], data[1])
predictions = decision_tree.predictResults(data[3], decisionTreeObj)
lastColumnResult = data[3][data[3].columns[-1:][0]]

#print("The decision tree is: " + str(decisionTreeObj))

#Showing the accuracy of the decision tree
accuracy = decision_tree.calculateAcc(lastColumnResult, predictions)
print ("Accuracy: " + repr(accuracy))
print ("Train: " + repr(len(data[4])))
print ("Test: "  + repr(len(data[3])))


#Decision Tree
print("\n\n---- Decision Tree - Soybean database ----\n")

#Reading file and setting up the correct database format
data = loadDatasetTree('databases/soybean.data', 0)

#Build the decision tree and testing it
decisionTreeObj = decision_tree.buildTree(data[0], data[1])
predictions = decision_tree.predictResults(data[3], decisionTreeObj)
lastColumnResult = data[3][data[3].columns[-1:][0]]

#print("The decision tree is: " + str(decisionTreeObj))

#Showing the accuracy of the decision tree
accuracy = decision_tree.calculateAcc(lastColumnResult, predictions)
print ("Accuracy: " + repr(accuracy))
print ("Train: " + repr(len(data[4])))
print ("Test: "  + repr(len(data[3])))


