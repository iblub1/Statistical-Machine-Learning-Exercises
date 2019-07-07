import numpy as np
import matplotlib.pyplot as plt
import os

# set the working directory
os.chdir("C:/Users/pistl/Desktop/SML/homework/homework3/")

# print the current working directory
os.getcwd()

# read in data 
def load_data():    
    data = np.loadtxt(fname = "C:/Users/pistl/Desktop/SML/homework/homework3/dataSets/iris.txt",delimiter=',', skiprows=0)
    return data

# calculate mean and standard_deviation
def calculate_stats(data):
    mean = np.mean(data[:,:4],axis=0)
    std_deviation = np.std(data[:,:4],axis=0)
    #eigen = np.linalg.eig(data)
    
    return mean, std_deviation

# normalize data to zero mean and unit variance
def normalize_data(data, mean, std_deviation):
    data_normalized = ( data[:,:4] - mean ) / std_deviation
    return data_normalized

# read in data 
data = load_data()

# calculate mean and standard_deviation
stats = calculate_stats(data)
mean = stats[0]
std_deviation = stats[1]

# normalize data to zero mean and unit variance
data_normalized = normalize_data(data,mean,std_deviation)