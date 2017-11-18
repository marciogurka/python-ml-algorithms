import numpy as np
import random

#init with a random board
def createBoard(N):
    X = np.array([(random.uniform(-1, 1), random.uniform(-1, 1)) for i in range(N)])
    return X

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
def groupClusters(dataValues, centroids):
    clusters  = {}
    for value in dataValues:
        bestCentroidKey = min([(i[0], np.linalg.norm(value-centroids[i[0]])) \
                         for i in enumerate(centroids)], key=lambda t:t[1])[0]
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


#Draw out the different clusters
def drawClusters(dataValues, centroids):
    import matplotlib.pyplot as plt
    #Choose a different colour for each cluster
    colour = plt.cm.rainbow(np.linspace(0,1,len(centroids)))
    plt.figure()
    plt.title("KMeans Results")
    for i, centroid in enumerate(centroids):
        #Grab just the samples fpr the given cluster and plot them out with a new colour
        samples = dataValues[i]
        for sample in samples:
            plt.scatter(sample[0], sample[1], c=colour[i])
        plt.plot(centroid[0], centroid[1], markersize=35, marker="x", color='k', mew=10)
        plt.plot(centroid[0], centroid[1], markersize=30, marker="x", color='m', mew=5)
    plt.savefig('test.png')


#setting a board, getting the result then plotting it.
vet = createBoard(300)
result = findCentroids(vet, 3)

drawClusters(result[1], result[0])

