import numpy as np
import math
import random

num_hidden_layer_nodes = 25

def readData():
    X = np.loadtxt('data/ex4data1.csv', delimiter = ',')
    y = np.loadtxt('data/ex4data2.csv', delimiter = ',')
    Theta1 = np.loadtxt('data/ex4theta1.csv', delimiter = ',')
    Theta2 = np.loadtxt('data/ex4theta2.csv', delimiter = ',')

    return (X, y, Theta1, Theta2)

def sigmoid(z):     #sigmoid function
    return (1/(1 + math.exp(-1 * z)))

def sigmoidGradient(z):     #sigmoid gradient function
    return (math.exp(-1 * z)/((1 + math.exp(-1 * z)) ** (2)))

def feedForward(X, theta1, theta2, all):
    z2 = np.dot(theta1, np.transpose(X))

    #Calculate hidden layer
    sig = np.vectorize(sigmoid)
    a2 = sig(z2)

    a2 = np.insert(a2, 0, 1)
    #a2 = [np.insert(i, 0, 1) for i in z2]   #add bias unit
    z3 = np.dot(theta2, a2)

    a3 = sig(z3)

    if (all == True):
        return (z2, a2, z3, a3)
    else:
        return a3

def costFunction(X, y, theta1, theta2, K, lambda1):
    m = len(X)
    yVector = []
    for i in y:     #y to vector form e.g. 1 becomes [0 1 0 ... 0]
        arr = [0] * K
        arr[int(i) - 1] = 1
        yVector.append(arr)

    i = 0
    sum = 0
    while (i < m):
        k = 0

        htheta = feedForward(X[i], theta1, theta2, False)
        while (k < K):
            term1 = (-1 * yVector[i][k]) * math.log(htheta[k])
            term2 = (1 - yVector[i][k]) * math.log(1 - htheta[k])
            sum = sum + (term1 - term2)

            k = k + 1
        i = i + 1

    J = (1.0/m) * sum

    #Regularization term
    sum1 = 0
    i = 0
    while (i < len(theta1)):        #theta 1 regularization
        j = 1   #skip bias term
        while (j < len(theta1[0])):
            sum1 = sum1 + (theta1[i][j] ** 2)
            j = j + 1
        i = i + 1

    i = 0
    while (i < len(theta2)):        #theta 2 regularization
        j = 1   #skip bias term
        while (j < len(theta2[0])):
            sum1 = sum1 + (theta2[i][j] ** 2)
            j = j + 1
        i = i + 1

    J = J + ((float(lambda1)/(2 * m)) * sum1)
    return J

def randInitialize(numRows, numCols):
    epsinit = 0.12
    vals = np.random.rand(numRows, numCols) * (2 * epsinit)
    theta = np.subtract(vals, np.ones((numRows, numCols)) * epsinit)
    return theta

def backPropagation(X, y, theta1, theta2, K, lambda1):
    m = len(X)
    yVector = []
    for i in y:     #y to vector form e.g. 1 becomes [0 1 0 ... 0]
        arr = [0] * K
        arr[int(i) - 1] = 1
        yVector.append(arr)

    gradSig = np.vectorize(sigmoidGradient)     #vectorize sigmoid Gradient function
    Delta_1 = np.zeros((len(theta1), len(theta1[0])))
    Delta_2 = np.zeros((len(theta2), len(theta2[0])))
    i = 0
    while (i < m):
        z2, a2, z3, a3 = feedForward(X[i], theta1, theta2, True)

        k = 0
        delta3 = []     #little in training example i
        while (k < K):
            delta3.append(a3[k] - yVector[i][k])
            k = k + 1

        #little delta 2 for training example i
        delta2 = np.multiply(np.dot(np.transpose(theta2)[1:], delta3), gradSig(z2))

        #Accumulate gradient from examples
        Delta_1 = np.add(Delta_1, np.outer(delta2, X[i]))
        Delta_2 = np.add(Delta_2, np.outer(delta3, a2))
        i = i + 1

    theta1_grad = (1.0/m) * Delta_1
    theta2_grad = (1.0/m) * Delta_2

    #regularization terms
    reg_term_1 = np.multiply(float(lambda1)/m, theta1)
    reg_term_2 = np.multiply(float(lambda1)/m, theta2)
    i = 0
    while (i < len(theta1)):    #first column should be zeros for reg term
        reg_term_1[i][0] = 0
        i = i + 1

    i = 0
    while (i < len(theta2)):    #first column should be zeros for reg term
        reg_term_2[i][0] = 0
        i = i + 1

    theta1_grad = np.add(theta1_grad, reg_term_1)
    theta2_grad = np.add(theta2_grad, reg_term_2)

    return (theta1_grad, theta2_grad)

def checkGradient(X, y, theta1, theta2, theta1_grad, theta2_grad):
    i = 0
    while (i < 1):        #gradient for theta1
        j = 0
        while (j < 10):
            eps = .001
            epsarray = np.zeros((len(theta1), len(theta1[0])))
            k = random.randint(0, len(theta1) - 1)
            q = random.randint(0, len(theta1[0]) - 1)
            epsarray[k, q] = eps

            gradApprox = (costFunction(X, y, np.add(theta1, epsarray), theta2, 10, 1) - costFunction(X, y, np.subtract(theta1, epsarray), theta2, 10, 1))/(2 * eps)
            print("Theta1_grad element %d, %d is %f while numerical grad element %d, %d, is %f" % (k, q, theta1_grad[k, q], k, q, gradApprox))
            j = j + 1
        i = i + 1

    i = 0
    while (i < 1):        #gradient for theta2
        j = 0
        while (j < 5):
            eps = .001
            epsarray = np.zeros((len(theta2), len(theta2[0])))
            k = random.randint(0, len(theta2) - 1)
            q = random.randint(0, len(theta2[0]) - 1)
            epsarray[k, q] = eps

            gradApprox = (costFunction(X, y, theta1, np.add(theta2, epsarray), 10, 1) - costFunction(X, y, theta1, np.subtract(theta2, epsarray), 10, 1))/(2 * eps)
            print("Theta1_grad element %d, %d is %f while numerical grad element %d, %d, is %f" % (k, q, theta2_grad[k, q], k, q, gradApprox))
            j = j + 1
        i = i + 1


if __name__ == "__main__":
    X, y, theta1, theta2 = readData()

    X = [np.insert(i, 0, 1) for i in X]     #Add bias column of 1s
    print(costFunction(X, y, theta1, theta2, 10, 1))

    #Train neural network
    theta1 = randInitialize(num_hidden_layer_nodes, len(X[0]) - 1)
    theta2 = randInitialize(10, num_hidden_layer_nodes)
    theta1 = [np.insert(i, 0, 1) for i in theta1]     #Add column of 1s
    theta2 = [np.insert(i, 0, 1) for i in theta2]     #Add column of 1s

    theta1_grad, theta2_grad = backPropagation(X, y, theta1, theta2, 10, 1)
    #checkGradient(X, y, theta1, theta2, theta1_grad, theta2_grad)
