import numpy as np

#Split the data using the Gain Information as a decider where to split
def recursiveSplit(x, y):
    if checkPurity(y) or len(y) == 0:
        return y
    gain = np.array([calculateGainInformation(y, x_attr) for x_attr in x.T])
    if np.all(gain < 1e-7):
        return y
    selectedAttribute = np.argmax(gain)
    sets = doPartition(x[:, selectedAttribute])
    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)
        res["x%d = %s" % (int(selectedAttribute), k)] = recursiveSplit(x_subset, y_subset)
    return res

#Calculate the entropy of the system
def calculateEntropy(value):
    result = 0
    val, counts = np.unique(value, return_counts=True)
    freqs = counts.astype('float')/len(value)
    for x in freqs:
        if x != 0.0:
            result -= x * np.log2(x)
    return result

#Separate the array creating a dict according to the keys
def doPartition(x):
    return {t: (x==t).nonzero()[0] for t in np.unique(x)}

#Calculate the Gain of Information of the attributes
def calculateGainInformation(y, x):
    res = calculateEntropy(y)
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)
    for p, v in zip(freqs, val):
        res -= p * calculateEntropy(y[x == v])
    return res

#Check the purity of an array
def checkPurity(array):
    return len(set(array)) == 1

#Predict some result using the decision tree built previously
def predict(item, decisionTree):
    for i in (decisionTree):
        aux = i.replace(" ", "").replace("x", "").split("=")
        if item[float(aux[0])] == float(aux[1]):
            if isinstance(decisionTree[i], dict):
                return predict(item, decisionTree[i])
            else:
                print('*** Decision Tree Prediction = ' + str(decisionTree[i][0]) + ', Real Value = ' + str(item[item.index[-1]]))
                return decisionTree[i][0]

#Calculate the accuracy of the predictions
def calculateAcc(realResults, predictionsResults):
    correct = 0
    for index, object in enumerate(realResults):
        if object == predictionsResults[index]:
            correct += 1
    return (correct/float(len(realResults))) * 100.0

#Build the decision tree
def buildTree(x, y):
    X = np.array(x).T
    return recursiveSplit(X, y)

#Predict function that is called on main.py
def predictResults(x, decisionTree):
    predictions = []
    for row in x.iterrows():
        predictions.append(predict(row[1],decisionTree))
    return predictions