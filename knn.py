import numpy as np
import operator

#Calculate the Euclidean Distance between 2 samples
def euclideanDistance(instance1, instance2, length):
    a = instance1[:]
    b = instance2[:]
    #removing the last char of the training obj (the class)
    if len(a) > length:
        a.pop()
    if len(b) > length:
        b.pop()
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    distance = np.linalg.norm(a-b, 2, 0)
    return distance

#Get the closest n neighbors of the samples
def getNeighbors(trainingSet, testObject, numberOfNeighbors, length):
    distances = []
    for i in range(len(trainingSet)):
        dist = euclideanDistance(testObject, trainingSet[i], length)
        distances.append((trainingSet[i], dist))
    distances.sort(key=lambda item: item[1])
    neighbors = []
    for x in range(numberOfNeighbors):
        neighbors.append(distances[x][0])
    return neighbors

#Get the prediction result based on the neighbors
def getResponse(neighbors):
    classGroupCount = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classGroupCount:
            classGroupCount[response] += 1
        else:
            classGroupCount[response] = 1
    groupCountSort = sorted(classGroupCount.items(), key=operator.itemgetter(1), reverse=True)
    return groupCountSort[0][0]

#Calculate the accuracy of the results based on the expected result
def calculateAcc(testSet, results):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == results[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
