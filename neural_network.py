import numpy as np

def setTrainingInput():
    return np.array([[1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])

def setTrainingOutput():
    return np.array([[1, 0, 1, 0, 1, 0]]).T

def executeNetworkTraining():
    trainingInput = setTrainingInput()
    trainingOutput = setTrainingOutput()

    np.random.seed(1)
    synapticWeights = 2 * np.random.random((4, 1)) - 1

    for iteration in range(100000):
        output = 1 / (1 + np.exp(-(np.dot(np.squeeze(trainingInput), synapticWeights))))
        synapticWeights += np.dot(trainingInput.T, (trainingOutput - output) * output * (1 - output))

    executeNetworkTest(synapticWeights)

def executeNetworkTest(weights):
    print ("Input: [1, 0, 0, 1]")
    print ("Neural Network prediction: " + repr((1 / (1 + np.exp(-(np.dot(np.array([1, 0, 0, 1]), weights)))))[0]))
    print ("\nInput: [1, 1, 0, 1]")
    print ("Neural Network prediction: " + repr((1 / (1 + np.exp(-(np.dot(np.array([1, 1, 0, 1]), weights)))))[0]))
    print ("\nInput: [1, 1, 1, 1]")
    print ("Neural Network prediction: " + repr((1 / (1 + np.exp(-(np.dot(np.array([1, 1, 1, 1]), weights)))))[0]))
    print ("\nInput: [0, 0, 0, 0]")
    print ("Neural Network prediction: " + repr((1 / (1 + np.exp(-(np.dot(np.array([0, 0, 0, 0]), weights)))))[0]))
    print ("\nInput: [1, 0, 0, 0]")
    print ("Neural Network prediction: " + repr((1 / (1 + np.exp(-(np.dot(np.array([1, 0, 0, 0]), weights)))))[0]))