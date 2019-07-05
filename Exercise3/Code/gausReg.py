import numpy as np
import matplotlib.pyplot as plt
import math


def gaussian(x,mu,h):
    return np.exp(-(x-mu)**2/(2*h**2))/(h*np.sqrt(2*np.pi))
  

# Load data
data = np.loadtxt("linRegData.txt")
split_row = 30
train = data[:split_row]
test = data[split_row:]

x_train = np.reshape(train, len(train)*2)[0::2]
y_train = np.reshape(train, len(train)*2)[1::2]
x_test = np.reshape(test, len(test)*2)[0::2]
y_test = np.reshape(test, len(test)*2)[1::2]

# Model Parameter
number_features = 20
mean = np.linspace(0,2, number_features)
h = math.sqrt(0.02)  # Std
ridge = 10**(-6)

# Create Design Matrix
Phi = np.zeros(shape=(len(x_train), number_features))
sums = np.zeros(len(x_train))
for i in range(number_features):
  Phi[:,i] = gaussian(x_train, mean[i], h)  # Use rbf to create PHI 

for i in range(len(x_train)): 
  sums[i] = np.sum(Phi[i,:])  # After Phi is filled, calculate sums of all rbf
for i in range(len(x_train)): 
  Phi[i,:] = np.divide(Phi[i,:], sums[i])  # Normalize so sum = 1
  
# Calculate left pseudo inverse
v = np.dot(np.transpose(Phi), Phi)
w = np.linalg.inv(v + ridge*np.identity(Phi.shape[1]))
phiPseu = np.dot(w, np.transpose(Phi))

# Model fitting
theta = np.dot(phiPseu, y_train)

# Model prediction
predictions = np.zeros(shape=(len(x_test)))
gaus = np.zeros(shape=(number_features))

for i in range(len(x_test)):
  for j in range(number_features):
    gaus[j] = gaussian(x_test[i], mean[j], h)  # Use Gaussian
  gaus = np.divide(gaus, np.sum(gaus))  # Normalize so sum =1
  predictions[i] = np.dot(gaus, theta)

# Evaluate RMSE
rmse = np.sqrt(np.mean((predictions-y_test)**2))
print(rmse)

# Plot Phi
plt.rcParams['figure.dpi']= 150
plt.xlabel("x")
plt.ylabel("Phi(x)")
axis = np.linspace(0,2, 1000)
Phi = np.zeros(shape=(len(axis), number_features))
sums = np.zeros(len(axis))
for i in range(number_features):
  Phi[:,i] = gaussian(axis, mean[i], h)  # Use rbf to create features
  
for i in range(len(axis)):  # Normalize features
  sums[i] = np.sum(Phi[i,:])
  Phi[i,:] = np.divide(Phi[i,:], sums[i]) 
  
for i in range(number_features):  # Plot Features
  plt.plot(axis, Phi[:,i])
  
