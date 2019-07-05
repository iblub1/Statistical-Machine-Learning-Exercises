import numpy as np

# Load Data
data = np.loadtxt("ldaData.txt")
c1 = data[:50]  # First 50 values belong to class 1
c2 = data[50:93]  # Next 43 to class 2
c3 = data[-44:]  # The last 44 to class 3

# Calculate prios
pC1 = len(data) / len(c1)
pC2 = len(data) / len(c2)
pC3 = len(data) / len(c3)

# Calculate mean and variance 
mu1 = np.mean(c1, axis=0)
mu1full = np.tile(mu1, (len(c1), 1))  # not in use. Can be used to check for probability if c1 belongs into c1
mu2 = np.mean(c2, axis=0)
mu2full = np.tile(mu2, (len(c2), 1))
mu3 = np.mean(c3, axis=0)
mu3full = np.tile(mu3, (len(c3), 1))

s1 = np.std(c1, ddof=1)
cov1 = np.cov(c1, rowvar=False)  # Rows = Observations, Column = Variable shapep should be 2x2
s2 = np.std(c2, ddof=1)
cov2 = np.cov(c2, rowvar=False)
s3 = np.std(c3, ddof=1)
cov3 = np.cov(c3, rowvar=False)


def multivariate_gaussian(x, mu, cov):
  i_cov = np.linalg.pinv(cov)  
  det_cov = np.linalg.det(cov)
  
  frac = 1 / np.sqrt((2*np.pi)**len(x) * det_cov)
  exp = np.exp(-1/2 * np.dot(np.dot((x-mu) , i_cov) , (x-mu).T))
  
  return frac * exp

# Empty lists to store points
listC1 = []
listC2 = []
listC3 = []
for i in range(len(data)):
  posC1 = multivariate_gaussian(data[i], mu1, cov1) * pC1
  posC2 = multivariate_gaussian(data[i], mu2, cov2) * pC2
  posC3 = multivariate_gaussian(data[i], mu3, cov3) * pC3

  if(posC1 > posC2 and posC1 > posC3):
    listC1.append(data[i])
  elif(posC2 > posC1 and posC2 > posC3):
    listC2.append(data[i])
  elif(posC3 > posC1 and posC3 > posC2):
    listC3.append(data[i])
  else:
    print("Couldn't classify point")
    
predictedC1 = np.array(listC1)  # Convert lists to np.arrays
predictedC2 = np.array(listC2)
predictedC3 = np.array(listC3)


# Plot the results
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi']= 150
plt.scatter(predictedC1[:,0], predictedC1[:,1], c="red")
plt.scatter(predictedC2[:,0], predictedC2[:,1], c="blue")
plt.scatter(predictedC3[:,0], predictedC3[:,1], c="green")