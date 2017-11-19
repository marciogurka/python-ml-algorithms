import csv
import random
import knn
import kmeans
import neural_network
import decision_tree
import pandas as pd
import numpy as np

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])



print ("\n\n---- KMeans - Libras database ----\n")
trainingSet=[]
testSet=[]
dataset = []
with open('databases/movement_libras_10.data', 'r') as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)
    for x in range(len(dataset)-1):
        for y in range(len(dataset[x])):
            dataset[x][y] = float(dataset[x][y])
k = 1
kmeans.calculateCentroids(dataset, k)
k = 3
kmeans.calculateCentroids(dataset, k)
k = 5
kmeans.calculateCentroids(dataset, k)
k = 7
kmeans.calculateCentroids(dataset, k)

print ("\n\n---- KNN - Iris database ----\n")
loadDataset('databases/iris.data', 0.66, trainingSet, testSet)
predictions=[]
length = 4
k = 3
for x in range(len(testSet)):
    neighbors = knn.getNeighbors(trainingSet, testSet[x], k, length)
    result = knn.getResponse(neighbors)
    predictions.append(result)
    print('*** KNN Prediction =' + repr(result) + ', Real Value=' + repr(testSet[x][-1]))
accuracy = knn.calculateAcc(testSet, predictions)

print ("Accuracy: " + repr(accuracy))
print ("Train: " + repr(len(trainingSet)))
print ("Test: "  + repr(len(testSet)))


print ("\n\n---- KNN - Soybean database ----\n")
trainingSet=[]
testSet=[]
loadDataset('databases/soybean.data', 0.66, trainingSet, testSet)
predictions=[]
length = 4
k = 3
for x in range(len(testSet)):
    neighbors = knn.getNeighbors(trainingSet, testSet[x], k, length)
    result = knn.getResponse(neighbors)
    predictions.append(result)
    print('*** KNN Prediction =' + repr(result) + ', Real Value=' + repr(testSet[x][-1]))
accuracy = knn.calculateAcc(testSet, predictions)

print ("Accuracy: " + repr(accuracy))
print ("Train: " + repr(len(trainingSet)))
print ("Test: "  + repr(len(testSet)))


print ("\n\n---- Neural Network ----\n")
neural_network.executeNetworkTraining()

print("\n\n---- Decision Tree ----\n")

#Reading file and setting up the correct database format
df = pd.read_csv('databases/soybean.data', header=None)

testSet = pd.DataFrame()
trainingSet = pd.DataFrame()
split = 0.66

for row in df.iterrows():
    if random.random() < split:
        trainingSet = trainingSet.append(row[1])
    else:
        testSet = testSet.append(row[1])

index_last_column = df.columns[-1]
last_column = trainingSet[trainingSet.columns[-1:][0]]
x = []
y = np.array(last_column)
for column in df:
    if column != index_last_column:
        x.append(trainingSet[column])

#Build the decision tree and testing it
decisionTreeObj = decision_tree.buildTree(x, y)
predictions = decision_tree.predictResults(testSet, decisionTreeObj)
lastColumnResult = testSet[testSet.columns[-1:][0]]

print("The decision tree is: " + str(result))

#Showing the accuracy of the decision tree
accuracy = decision_tree.calculateAcc(lastColumnResult, predictions)
print ("Accuracy: " + repr(accuracy))
print ("Train: " + repr(len(trainingSet)))
print ("Test: "  + repr(len(testSet)))
