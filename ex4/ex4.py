import numpy as np
import math
import random
from scipy.optimize import fmin_cg

num_input_size = 400
num_hidden_layer_nodes = 25
num_output_size = 10

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

def costFunction(theta, X, y, K, lambda1):
    theta1, theta2 = unflattenParams(theta)

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

def backPropagation(theta, X, y, K, lambda1):
    theta1, theta2 = unflattenParams(theta)

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

    return flattenParams(theta1_grad, theta2_grad)

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

            gradApprox = (costFunction(flattenParams(np.add(theta1, epsarray), theta2), X, y, 10, 1) - costFunction(flattenParams(np.subtract(theta1, epsarray), theta2), X, y, 10, 1))/(2 * eps)
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

            gradApprox = (costFunction(flattenParams(theta1, np.add(theta2, epsarray)), X, y, 10, 1) - costFunction(flattenParams(theta1, np.subtract(theta2, epsarray)), X, y, 10, 1))/(2 * eps)
            print("Theta1_grad element %d, %d is %f while numerical grad element %d, %d, is %f" % (k, q, theta2_grad[k, q], k, q, gradApprox))
            j = j + 1
        i = i + 1


def flattenParams(theta1, theta2):
    #flattened params
    flattenned_theta1 = np.array(theta1).flatten()
    flattenned_theta2 = np.array(theta2).flatten()
    flattenned_theta = np.append(flattenned_theta1, flattenned_theta2)
    return flattenned_theta

def unflattenParams(theta):
    theta1_size = (num_hidden_layer_nodes * (num_input_size + 1))
    theta2_size = (num_output_size * (num_hidden_layer_nodes + 1))

    theta1 = np.reshape(theta[0:theta1_size], (num_hidden_layer_nodes, num_input_size + 1))
    theta2 = np.reshape(theta[theta1_size:theta1_size+theta2_size], (num_output_size, num_hidden_layer_nodes + 1))
    return (theta1, theta2)

def predict(X, theta1_opt, theta2_opt, y):
    numCorrect = 0
    numWrong = 0

    i = 0
    while(i < len(X)):
        ans = feedForward(X[i], theta1_opt, theta2_opt, False)

        maxval = -1
        maxindex = -1
        j = 0
        while (j < len(ans)):
            if (ans[j] > maxval):
                maxval = ans[j]
                maxindex = j

            j = j + 1
        if (y[i] == (maxindex + 1)):
            numCorrect = numCorrect + 1
        else:
            numWrong = numWrong + 1
        i = i + 1

    print("Percentage of classifications correct is: %f" % (float(numCorrect)/(numCorrect + numWrong)))

def callbackfunc(x):
    print("hi")

if __name__ == "__main__":
    X, y, theta1, theta2 = readData()

    X = [np.insert(i, 0, 1) for i in X]     #Add bias column of 1s
    print(costFunction(flattenParams(theta1, theta2), X, y, 10, 1))

    #Train neural network
    theta1 = randInitialize(num_hidden_layer_nodes, len(X[0]) - 1)
    theta2 = randInitialize(10, num_hidden_layer_nodes)
    theta1 = [np.insert(i, 0, 1) for i in theta1]     #Add column of 1s
    theta2 = [np.insert(i, 0, 1) for i in theta2]     #Add column of 1s

    theta_grad = backPropagation(flattenParams(theta1, theta2), X, y, 10, 1)
    theta1_grad, theta2_grad = unflattenParams(theta_grad)
    #checkGradient(X, y, theta1, theta2, theta1_grad, theta2_grad)

    #optimize using fmin
    thetaOpt = fmin_cg(costFunction, flattenParams(theta1, theta2), backPropagation, args=(X,y,10,1), callback=callbackfunc, maxiter=50)
    theta1_opt, theta2_opt = unflattenParams(thetaOpt)

    #predict Percentage correct
    predict(X, theta1_opt, theta2_opt, y)

