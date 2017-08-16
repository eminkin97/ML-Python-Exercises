import csv
import numpy as np
import math

def readCSVData():
	X = []
	y = []

	with open('data/ex3data1.csv', 'r') as csvfile1:
		reader = csv.reader(csvfile1)
		for row in reader:
			X.append([float(i) for i in row])
	
	with open('data/ex3data2.csv', 'r') as csvfile2:
		reader = csv.reader(csvfile2)
		for row in reader:
			y.append(float(row[0]))

	X = np.array(X)
	y = np.array(y)
	
	return (X, y)

def sigmoid(z):
	return (1/(1 + math.exp(-1 * z)))

def unvectorizedCostFunction(theta, lam):
	#theta is a vector
	J = 0
	m = len(X)

	i = 0
	while (i < m):
		value = np.dot(X[i], theta)

		J = J + ((-1 * y[i] * math.log(sigmoid(value))) - ((1-y[i]) * math.log(1-sigmoid(value))))
		i = i + 1

	J = J/m

	#Adding Regularization
	J = J + ((lam/(2 * m)) * sum([a**2 for a in theta]))

	return J

		
def vectorizedCostFunction(theta, lam):
	#theta is vector
	J = 0
	Xtheta = np.dot(X, theta)
	
	sum1 = sum(np.multiply([-1 * a for a in y], [math.log(sigmoid(a)) for a in Xtheta]))	#First part of sum in linear regression cost function
	sum2 = sum(np.multiply([a-1 for a in y], [math.log(1 - sigmoid(a)) for a in Xtheta]))	#Second part of sum in linear regression cost function

	J = (1/len(Xtheta)) * (sum1 + sum2)

	#Adding Regularization
	J = J + ((lam/(2 * len(Xtheta))) * sum([a**2 for a in theta]))

	return J

def vectorizedGradient(theta, lam):
	Xtheta = np.dot(X, theta)
	m = len(Xtheta)

	dJdtheta = (1/m) * np.dot(X.transpose(), np.subtract([sigmoid(a) for a in Xtheta], y))

	#Adding Regularization
	theta[0] = 0	#first gradient doesnt get regularization term
	dJdtheta = np.add(dJdtheta, [(lam * a)/m for a in theta])
	return dJdtheta

def OneVsAllMinimization():
	

if  __name__ == "__main__":
	[X, y] = readCSVData()
	theta = np.array([0] * 400)
	print(unvectorizedCostFunction(theta, 5))
	print(vectorizedCostFunction(theta, 5))
	print(vectorizedGradient(theta, 5))
