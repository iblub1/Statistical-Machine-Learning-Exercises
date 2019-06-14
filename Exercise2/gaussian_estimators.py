# import necessary Python packages
import os
import numpy as np

# set the working directory
os.chdir("C:/Users/pistl/Desktop/hw2/")

# print the current working directory
os.getcwd()

# load data
densEst1 = np.loadtxt(fname = "C:/Users/pistl/Desktop/hw2/dataSets/densEst1.cvs")
densEst2 = np.loadtxt(fname = "C:/Users/pistl/Desktop/hw2/dataSets/densEst2.cvs")

def sample_mean(data):
    N = np.size(data,0)
    column_sums = np.sum(data, axis=0)
    means = 1/N * column_sums
    return means

def sample_variance_biased(data,means):
    N = np.size(data,0)
    x = data - means
    variances_biased = 1/N * np.matmul(np.transpose(x),x)
    return variances_biased

def sample_variance_unbiased(data,means):
    N = np.size(data,0)
    denominator = N - 1
    x = data - means
    variances_biased = 1/denominator * np.matmul(np.transpose(x),x)
    return variances_biased

print('ESTIMATORS OF DATASET densEst1\n')
means = sample_mean(densEst1)
print('means\n',means)
print('\nvariance_biased\n',sample_variance_biased(densEst1,means))
print('\nvariance_unbiased\n',sample_variance_unbiased(densEst1,means))

print('--------------------------------------')

print('ESTIMATORS OF DATASET densEst2\n')
means = sample_mean(densEst2)
print('means\n',means)
print('\nvariance_biased\n',sample_variance_biased(densEst2,means))
print('\nvariance_unbiased\n',sample_variance_unbiased(densEst2,means))
