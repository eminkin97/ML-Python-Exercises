import numpy as np
#import matplotlib.pyplot as plt

iterations = 1500
alpha = 0.01

def readData():
    data = np.loadtxt("ex1data1.txt", delimiter=",")
    plotData(data)

    X = [i[:-1] for i in data]
    y = [i[1] for i in data]

    return (X, y)

def plotData(data):
    print("hi")

    #plt.plot([i[0] for i in data], [i[1] for i in data], 'ro')
    #plt.xlabel("Population of city in 10,000's")
    #plt.ylabel("Profit in $10,000's")

def gradientDescent(initialtheta, X, y):
    i = 0
    m = len(X)
    theta = initialtheta

    while (i < iterations):
        j = 0
        temp0 = 0
        temp1 = 0

        while (j < m):
            temp0 = temp0 + (np.dot(theta, X[j]) - y[j]) * X[j][0]
            temp1 = temp1 + (np.dot(theta, X[j]) - y[j]) * X[j][1]
            j = j + 1

        #Simultaneous update
        theta[0] = theta[0] - (alpha/m) * temp0
        theta[1] = theta[1] - (alpha/m) * temp1

        print(computeCost(theta, X, y))

        i = i + 1



def computeCost(theta, X, y):
    m = len(X)

    i = 0
    sum1 = 0
    while (i < m):
        sum1 = sum1 + (np.dot(theta, X[i]) - y[i])**2
        i =  i + 1

    J = (1/(2*m)) * sum1

    return J

if __name__ == "__main__":
    [X, y] = readData()

    X = [np.insert(i, 0, 1) for i in X]    #add column of 1s to X to represent theta 0

    theta = [0, 0]
    print(computeCost(theta, X, y))
    gradientDescent(theta, X, y)
