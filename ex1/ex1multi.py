import numpy as np

iterations = 50
alpha = 1

def readData():
    data = np.loadtxt("ex1data2.txt", delimiter=",")

    data = featureNormalize(data)
    y = [a[-1] for a in data]
    X = [a[:-1] for a in data]

    return (X, y)

def gradientDescent(initialtheta, X, y):
    i = 0
    m = len(X)  #Number of data sets to train from
    z = len(X[0])   #Number of columns for an entry
    theta = initialtheta

    while (i < iterations):
        j = 0
        temp = [0] * z

        while (j < m):
            k = 0
            while (k < z):
                temp[k] = temp[k] + (np.dot(theta, X[j]) - y[j]) * X[j][k]
                k = k + 1

            j = j + 1

        #Simultaneous update
        k = 0
        while (k < len(theta)):
            theta[k] = theta[k] - (alpha/m) * temp[k]
            k = k + 1

        print(computeCost(theta, X, y))

        i = i + 1
    return theta    #return value of theta

def computeCost(theta, X, y):
    m = len(X)

    i = 0
    sum1 = 0
    while (i < m):
        sum1 = sum1 + (np.dot(theta, X[i]) - y[i])**2
        i =  i + 1

    J = (1/(2*m)) * sum1

    return J

def featureNormalize(data):
    k = 0
    means = []
    stdevs = []
    while (k < len(data[0])):       #Calculates means and standard deiations
        means.append(np.mean([i[k] for i in data]))
        stdevs.append(np.std([i[k] for i in data]))
        k = k + 1


    for i in data:      #normalize features
        j = 0
        while (j < len(i)):
            i[j] = (i[j] - means[j])/stdevs[j]
            j =  j + 1

    return data

def normalEquationsOption(X, y):
    a = np.dot(np.transpose(X), X)
    b = np.linalg.inv(a)
    c = np.dot(b, np.transpose(X))
    theta = np.dot(c, y)
    print(theta)

if __name__ == "__main__":
    [X, y] = readData()

    X = [np.insert(i, 0, 1) for i in X]    #add column of 1s to X to represent theta 0

    theta = [0, 0, 0]
    print(computeCost(theta, X, y))
    gradientDescent(theta, X, y)
    print(theta)
    print("\nNormalEquations: ")
    normalEquationsOption(X, y)
