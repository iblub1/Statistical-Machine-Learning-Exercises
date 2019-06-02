import numpy as np
import matplotlib.pyplot as plt

def rosen(x):
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)

def rosen_der(x):
    """
    	Gradient of the Rosenbrock function of for gradient descent.
    """
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

cur_x = np.random.normal(0, 0.1, 20)  # Start point randomly drawn from gausian mu = 0, sigma = 0.1
learning_rate = 0.0001  # Learning rate

max_iteration = 10000  # maximum number of iterations
history = np.zeros((max_iteration+1, 2))

for k in range(max_iteration):
    prev_x = cur_x  # Store current x value in prev_x
    cur_x = prev_x - learning_rate * rosen_der(prev_x)  # Grad descent
    error = abs(cur_x - prev_x)
    k = k + 1  # iteration count

    history[k, :] = np.linalg.norm(error)

    # print("Iteration", iteration, "\nX value is", cur_x)  # Print iterations
    print(" Iteration: ", k, " \n")
    print(" Gradient: ", rosen_der(prev_x), "\n")
    print(" Error: ", error, "\n\n")

print("The local minimum occurs at", cur_x)


plt.plot(history)
plt.title("Error for LR: 0.0001")
plt.ylabel("Error")
plt.xlabel("Iteration")
plt.show()
