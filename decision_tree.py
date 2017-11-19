import numpy as np

def partition(a):
    return {c: (a==c).nonzero()[0] for c in np.unique(a)}

def entropy(s):
    res = 0
    val, counts = np.unique(s, return_counts=True)
    freqs = counts.astype('float')/len(s)
    for p in freqs:
        if p != 0.0:
            res -= p * np.log2(p)
    return res

def mutual_information(y, x):

    res = entropy(y)

    # We partition x, according to attribute values x_i
    val, counts = np.unique(x, return_counts=True)
    freqs = counts.astype('float')/len(x)

    # We calculate a weighted average of the entropy
    for p, v in zip(freqs, val):
        res -= p * entropy(y[x == v])

    return res


def is_pure(s):
    return len(set(s)) == 1

def recursive_split(x, y):
    # If there could be no split, just return the original set
    if is_pure(y) or len(y) == 0:
        return y

    # We get attribute that gives the highest mutual information
    gain = np.array([mutual_information(y, x_attr) for x_attr in x.T])
    selected_attr = np.argmax(gain)

    # If there's no gain at all, nothing has to be done, just return the original set
    if np.all(gain < 1e-6):
        return y


    # We split using the selected attribute
    sets = partition(x[:, selected_attr])

    res = {}
    for k, v in sets.items():
        y_subset = y.take(v, axis=0)
        x_subset = x.take(v, axis=0)

        res["x%d = %s" % (int(selected_attr), k)] = recursive_split(x_subset, y_subset)

    return res

def predict(item, decisionTree):
    for i in (decisionTree):
        aux = i.replace(" ", "").replace("x", "").split("=")
        if item[float(aux[0])] == float(aux[1]):
            if isinstance(decisionTree[i], dict):
                return predict(item, decisionTree[i])
            else:
                print('*** Decision Tree Prediction = ' + str(decisionTree[i][0]) + ', Real Value = ' + str(item[item.index[-1]]))
                return decisionTree[i][0]


def calculateAcc(realResults, predictionsResults):
    correct = 0
    for index, object in enumerate(realResults):
        if object == predictionsResults[index]:
            correct += 1
    return (correct/float(len(realResults))) * 100.0

def buildTree(x, y):
    X = np.array(x).T
    return recursive_split(X, y)

def predictResults(x, decisionTree):
    predictions = []
    for row in x.iterrows():
        predictions.append(predict(row[1],decisionTree))
    return predictions