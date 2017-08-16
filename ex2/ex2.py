import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
import math

def readData():
	data = np.loadtxt("ex2data1.txt", delimiter=",")
	return data

def plotData(data, decisionboundary, pt1, pt2):
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
	
	if (decisionboundary == True):
		plt.plot(pt1, pt2, color='salmon')

	plt.legend()
	plt.show()

def getDecisionBoundaryLine(X, y, theta):
	minval = 1000
	maxval = 0
	minindex = -1
	maxindex = -1

	i = 0
	while (i < len(X)):
		if (X[i][1] < minval):
			minval = X[i][1]
			minindex = i
		if (X[i][1] > maxval):
			maxval = X[i][1]
			maxindex = i
		i = i + 1

	xs = [X[minindex][1], X[maxindex][1]]
	print(xs)
	ys = [(1.0/theta[2]) * -1 * (theta[0] + theta[1] * xs[0]), (1.0/theta[2]) * -1 * (theta[0] + theta[1] * xs[1])]
	print(ys)
	return (xs, ys)


def sigmoid(x):	#sigmoid function
	return 1/(1 + math.exp(-1 * x))

def costFunction(theta, X, y):
	#cost Function for logistic regression
	m = len(X)

	i = 0
	sum1 = 0
	while (i < m):
		htheta = sigmoid(np.dot(X[i], theta))
		sum1 = sum1 + ((-1 * y[i] * math.log(htheta)) - ((1 - y[i]) * math.log(1 - htheta)))

		i = i + 1

	J = (1.0/m) * sum1
	return J

def gradient(theta, X, y):
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

		gradient.append((1.0/m) * sum1)
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
	plotData(data, False, 0, 0)

	X = [np.insert(a[:-1], 0, 1) for a in data]
	y = [a[-1] for a in data]

	theta = [0] * 3		#initial guess for theta

	print(costFunction(theta, X, y))
	print(gradient(theta, X, y))

	#optimize using fmin
	thetaOpt = fmin(costFunction, theta, args=(X,y), maxiter=400)
	print(thetaOpt)

	#plot Data with decision boundary
	point1, point2 = getDecisionBoundaryLine(X, y, thetaOpt)
	plotData(data, True, point1, point2)

	#predictions
	predict(X, y, thetaOpt)
