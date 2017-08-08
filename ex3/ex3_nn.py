import numpy as np
import math

def readData():
    X = np.loadtxt('data/ex3data1.csv', delimiter = ',')
    y = np.loadtxt('data/ex3data2.csv', delimiter = ',')
    Theta1 = np.loadtxt('data/ex3weights1.csv', delimiter = ',')
    Theta2 = np.loadtxt('data/ex3weights2.csv', delimiter = ',')

    return (X, y, Theta1, Theta2)

def sigmoid(z):     #sigmoid function
    return (1/(1 + math.exp(-1 * z)))

def predict(X, theta1, theta2):
    z2 = np.dot(theta1, np.transpose(X))

    #Calculate hidden layer
    sig = np.vectorize(sigmoid)
    a2 = sig(z2)

    a2 = np.insert(a2, 0, np.ones(5000), axis = 0)
    #a2 = [np.insert(i, 0, 1) for i in z2]   #add bias unit
    z3 = np.dot(theta2, a2)

    a3 = sig(z3)
    return a3

def numberCorrect(htheta, y):
    numCorrect = 0
    numWrong = 0

    i = 0
    while(i < len(htheta[0])):
        maxval = -1
        maxindex = -1

        j = 0
        while (j < len(htheta)):
            if (htheta[j][i] > maxval):
                maxval = htheta[j][i]
                maxindex = j
            j = j + 1

        if ((maxindex + 1) == y[i]):
            numCorrect = numCorrect + 1
        else:
            numWrong = numWrong + 1

        i = i + 1

    print (float(numCorrect)/(numCorrect + numWrong))

if __name__ == "__main__":
    X, y, theta1, theta2 = readData()

    X = [np.insert(i, 0, 1) for i in X]     #Add bias column of 1s
    htheta = predict(X, theta1, theta2)
    numberCorrect(htheta, y)
