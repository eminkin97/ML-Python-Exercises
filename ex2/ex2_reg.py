import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import math

def readData():
	data = np.loadtxt("ex2data2.txt", delimiter=",")
	return data

def plotData(data):
	admitted = []	#data sets with entries that were admitted
	notadmitted = []	#data sets with entries that were not admitted

	for i in data:
		if (i[2] == 0):
			#not admitted
			notadmitted.append(i)
		else:
			#admitted
			admitted.append(i)

	plt.plot([i[0] for i in admitted], [i[1] for i in admitted], 'ro', label="admitted")
	plt.plot([i[0] for i in notadmitted], [i[1] for i in notadmitted], 'bs', label="not admitted")

	plt.legend()
	plt.show()

def mapFeature(x1col, x2col):
    """ 
    Function that takes in a column of n- x1's, a column of n- x2s, and builds
    a n- x 28-dim matrix of featuers as described in the homework assignment
    """
    degrees = 6
    out = np.ones( (x1col.shape[0], 1) )

    for i in range(1, degrees+1):
        for j in range(0, i+1):
            term1 = x1col ** (i-j)
            term2 = x2col ** (j)
            term  = (term1 * term2).reshape( term1.shape[0], 1 ) 
            out   = np.hstack(( out, term ))
    return out


def sigmoid(x):	#sigmoid function
	return 1/(1 + math.exp(-1 * x))

def costFunction(theta, X, y, lambda1):
	#cost Function for logistic regression
	m = len(X)

	i = 0
	sum1 = 0
	while (i < m):
		htheta = sigmoid(np.dot(X[i], theta))
		sum1 = sum1 + ((-1 * y[i] * math.log(htheta)) - ((1 - y[i]) * math.log(1 - htheta)))

		i = i + 1

	#regularization term
	i = 0
	sum2 = 0
	while (i < len(theta)):
		sum2 = sum2 + (theta[i] ** 2)
		i = i + 1

	#put terms together
	J = ((1.0/m) * sum1) + ((lambda1/(2 * m)) * sum2)
	return J

def gradient(theta, X, y, lambda1):
	#compute the gradient for logistic regression
	gradient = []	#gradients for j = 0,..,n
	m = len(X)

	i = 0
	while (i < len(theta)):
		j = 0
		sum1 = 0.0

		while (j < m):	#calculates gradient for theta(i)
			htheta = sigmoid(np.dot(X[j], theta))

			sum1 = sum1 + ((htheta - y[j]) * X[j][i])
			j = j + 1


		if (i == 0):
			gradient.append((1.0/m) * sum1)
		else:
			gradient.append(((1.0/m) * sum1) + ((lambda1/m)	* theta[i]))

		i = i + 1

	return gradient

def predict(X, y, thetaOpt):
	numPredictionsCorrect = 0
	numPredictionsWrong = 0

	i = 0
	while (i < len(X)):
		htheta = sigmoid(np.dot(X[i], thetaOpt))
	
		if (htheta >= 0.5):
			prediction = 1
		else:
			prediction = 0

		if (prediction == y[i]):
			numPredictionsCorrect = numPredictionsCorrect + 1
		else:
			numPredictionsWrong = numPredictionsWrong + 1
		i = i + 1

	percentCorrect = float(numPredictionsCorrect)/(numPredictionsWrong + numPredictionsCorrect)
	print("Percent of predictions correct is %f" % percentCorrect)



if __name__ == "__main__":
	data = readData()
	plotData(data)

	X1 = [np.insert(a[:-1], 0, 1) for a in data]
	y = [a[-1] for a in data]
	X = mapFeature(np.array([a[1] for a in X1]), np.array([a[2] for a in X1]))	#maps to 28 features
	print(X)

	theta = [0] * 28	#initial guess for theta
	print(costFunction(theta, X, y, 1))
	print(gradient(theta, X, y, 1))

	#optimize using fmin
	thetaOpt = fmin(costFunction, theta, args=(X,y,1), maxiter=80000)
	print(thetaOpt)

	predict(X, y, thetaOpt)
