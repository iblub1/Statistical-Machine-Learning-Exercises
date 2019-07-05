import numpy as np

# Load data
data = np.loadtxt("linRegData.txt")
split_row = 20
train = data[:split_row]
test = data[split_row:]

x_train = np.reshape(train, len(train)*2)[0::2]
y_train = np.reshape(train, len(train)*2)[1::2]
x_test = np.reshape(test, len(test)*2)[0::2]
y_test = np.reshape(test, len(test)*2)[1::2]

# Model Parameter
degree = 21
ridge = 10**(-6)


# Create Design Matrix
Phi = np.zeros(shape=(len(x_train), degree+1))
for i in range(degree+1):
  Phi[:,i] = np.power(x_train, i)

# Calculate left pseudo inverse
#phiPseu = np.linalg.pinv(Phi)  # Use this line to let np calculate the pseudo inverse

ridgePhi = np.dot(Phi.T, Phi) + ridge*np.identity(Phi.shape[1])
phiPseu = np.dot(np.linalg.pinv(ridgePhi),  Phi.T)


# Model fitting
theta = np.dot(phiPseu, y_train)

# Model prediction
predictions = np.zeros(shape=(len(x_test)))
polynom = np.zeros(shape=(degree+1))

for i in range(len(x_test)):
  for j in range(degree+1):
    polynom[j] = np.power(x_test[i], j)
  predictions[i] = np.dot(polynom, theta)

# Evaluate RMSE
rmse = np.sqrt(np.mean((predictions-y_test)**2))
print(rmse)

