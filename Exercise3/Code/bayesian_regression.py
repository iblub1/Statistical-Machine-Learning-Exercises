import numpy as np
import matplotlib.pyplot as plt
import os

# set the working directory
os.chdir("C:/Users/pistl/Desktop/SML/homework/homework3/")

# print the current working directory
os.getcwd()

# Load data
data = np.loadtxt(fname = "C:/Users/pistl/Desktop/SML/homework/homework3/dataSets/linRegData.txt", skiprows=0)
split_row = 150
train = data[:split_row]
test = data[split_row:]

x_train = np.reshape(train, len(train)*2)[0::2]
y_train = np.reshape(train, len(train)*2)[1::2]
x_test = np.reshape(test, len(test)*2)[0::2]
y_test = np.reshape(test, len(test)*2)[1::2]

# Model Parameter
degree = 10
xaxis = np.linspace(0,2,100)
yaxis = np.linspace(0,2,100)

# Create Design Matrix for training data
phi = np.zeros(shape=(len(x_train), degree+1))
for i in range(degree+1):
  phi[:,i] = np.power(x_train, i)
  
# Create Design Matrix for plot
phi_axis = np.zeros(shape=(len(xaxis), degree+1))
for i in range(degree+1):
  phi_axis[:,i] = np.power(xaxis, i)

def calculate_predictive(phi):
    lam = 1e-6
    beta = 1 / 0.0025
    alpha = lam * beta
    phi_trans = np.transpose(phi)
    phi_squared = np.matmul(phi_trans,phi)
    identity=np.eye(degree+1)
    # calculate mu
    brackets_mu = np.linalg.inv(alpha / beta * identity + phi_squared)
    mu_1 = np.matmul(np.matmul(phi,brackets_mu),phi_trans)
    mu = np.matmul(mu_1,y_train) 
    # calculate cov
    brackets_cov = np.linalg.inv(alpha * identity + beta * phi_squared)
    cov = 1/beta * np.matmul(np.matmul(phi,brackets_cov),phi_trans)
    std_deviation = np.sqrt(np.diag(cov))
    std_deviation_2 = 2*std_deviation
    return mu, std_deviation_2

def calculate_predictive_axis(phi,yaxis):
    lam = 1e-6
    beta = 1 / 0.0025
    alpha = lam * beta
    phi_trans = np.transpose(phi)
    phi_squared = np.matmul(phi_trans,phi)
    identity=np.eye(degree+1)
    # calculate mu
    brackets_mu = np.linalg.inv(alpha / beta * identity + phi_squared)
    mu_1 = np.matmul(np.matmul(phi,brackets_mu),phi_trans)
    mu = np.matmul(mu_1,yaxis) 
    # calculate cov
    brackets_cov = np.linalg.inv(alpha * identity + beta * phi_squared)
    cov = 1/beta * np.matmul(np.matmul(phi,brackets_cov),phi_trans)
    std_deviation = np.sqrt(np.diag(cov))
    std_deviation_2 = 2*std_deviation
    return mu, std_deviation_2
    
mu=calculate_predictive(phi)[0]
std_deviation_2=calculate_predictive(phi)[1]

fig, ax = plt.subplots()
plt.scatter(x_train, sums)
plt.savefig('bayesian_150.png',dpi=300)
