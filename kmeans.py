import numpy as np
import random

def calculateCentroids(samplesSet, K):
    #samplesSet = np.array(samplesSet[:50])
    newCentroids = random.sample(list(samplesSet), K)
    oldCentroids = random.sample(list(samplesSet), K)
    while not hasConverged(newCentroids, oldCentroids):
        oldCentroids = newCentroids
        clusters = groupClusters(samplesSet, newCentroids)
        newCentroids = updateCentroids(clusters)

    calculateError(clusters, newCentroids, K)

#Find the centroids and return the groups and centroids values
def findCentroids(data, K):
    oldCentroids = random.sample(list(data), K)
    newCentroids = random.sample(list(data), K)
    while not hasConverged(newCentroids, oldCentroids):
        oldCentroids = newCentroids
        clusters = groupClusters(data, newCentroids)
        newCentroids = updateCentroids(clusters)
    return (newCentroids, clusters)

#Check if the centroids has changed
def hasConverged(newCentroids, oldCentroids):
    return (set([tuple(a) for a in newCentroids]) == set([tuple(a) for a in oldCentroids]))

#grouping the clusters with the nearest centroid
def groupClusters(dataValues, centroids ):
    clusters  = {}

    a = np.array(dataValues)
    b = np.array(centroids)

    data1 = a.astype(float)
    data2 = b.astype(float)

    for value in data1:
        bestCentroidKey = min([(i[0], np.linalg.norm(value - data2[i[0]])) for i in enumerate(data2)], key=lambda t:t[1])[0]
        try:
            clusters[bestCentroidKey].append(value)
        except KeyError:
            clusters[bestCentroidKey] = [value]
    return clusters

#Setting the new centroids value
def updateCentroids(clusters):
    newCentroids = []
    keys = sorted(clusters.keys())
    for k in keys:
        newCentroids.append(np.mean(clusters[k], axis = 0))
    return newCentroids

def calculateError(cluster, centroids, K):
    deviation = []
    deviation2 = []
    sse = []
    for i in range(len(cluster)):
        totalClusters = len(cluster[i])
        euclidianValue = []
        for j in range(len(cluster[i])):
            euclidianValue.append(np.linalg.norm(cluster[i][j] - centroids[i]))
        error = np.sum(euclidianValue)
        mean = error/totalClusters

        for j in range(len(cluster[i])):
            deviationValue = (np.linalg.norm(cluster[i][j] - centroids[i])) - mean
            deviation.append(deviationValue)
            deviation2.append(np.square(deviationValue))
        sse.append(np.sum(deviation2[i]))

        #print("SSE of cluster nº " + str(i + 1))
        #print(sse[i])
        #print("\n")
    print("SSE of the system: " + repr(np.sum(sse)) + " with k = " + str(K))