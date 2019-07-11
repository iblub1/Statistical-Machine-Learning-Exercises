import numpy as np
import matplotlib.pyplot as plt

def kernel(a, b):
    l = 0.1  # Set parameter
    squared_distance = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-.5 * (1 / l) * squared_distance)


# Unkown target function (used to generate data)
f = lambda x: (np.sin(x) + np.sin(x) ** 2).flatten()
s = 0.001  # Noise variance


fig = plt.figure()
for N in range(16):
    x_train = np.linspace(0, 5, N).reshape(-1, 1)  # Increase N to generate more training points at maximum uncertainity
    y = f(x_train) + s * np.random.randn(N)  # Generate noisy train data

    # Calculate kernel on train data
    K = kernel(x_train, x_train)
    L = np.linalg.cholesky(K + s * np.eye(N))  # Get square-root of matrix with cholesky

    # Points at which we want to predict the function
    n = 1256  # calculate from the given intervall
    x_test = np.linspace(0, 2 * np.pi, n).reshape(-1, 1)

    # Compute kernel on train test similarity
    K_s = kernel(x_train, x_test)

    # compute the mean for the test points. Equation systems is used instead of naively inversing K
    alpha = np.linalg.solve(L, K_s)

    mu = np.dot(alpha.T, np.linalg.solve(L, y))
    mu2 = mu.reshape(-1, 1)

    # Compute kernel on test data
    K_ss = kernel(x_test, x_test)

    # compute the variance for the  test points.
    h2 = np.diag(K_ss) - np.sum(alpha ** 2, axis=0)
    h = np.sqrt(h2)


    # Get posterior for test points
    L_ss = np.linalg.cholesky(K_ss + 1e-6 * np.eye(len(x_test)) - np.dot(alpha.T, alpha))
    f_pos = mu.reshape(-1, 1) + np.dot(L_ss, np.random.normal(size=(len(x_test), 1)))

    """ Plot the results"""
    ax = fig.add_subplot(4, 4, N+1)
    ax.plot(x_train, y, 'bs', ms=8)  # Plot obersvations
    try:
        ax.fill_between(x_test.flat, mu-2*h, mu+2*h, color="#dddddd")  # Plot sigma
    except:
        print("Couldnt fill empty space")  # Normally this shouldn't matter

    ax.plot(x_test, mu, 'r--', lw=2)  # Plot mu

    #plt.plot(x_test, f_pos)  # Plot the posterior

    plt.plot(x_test, f(x_test))  # Plot True function

    title_string = "Number of observations: " + str(N)
    plt.title(title_string)

plt.show()
