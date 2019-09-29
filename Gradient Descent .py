import numpy as np
import matplotlib.pylab as plt

plot_x = np.linspace(-1, 6, 141)
print(plot_x)

plot_y = (plot_x - 2.5) ** 2 - 1

plt.plot(plot_x, plot_y)
plt.show()


def dJ(theta):
    return 2 * (theta - 2.5)


def J(theta):
    return (theta - 2.5) ** 2 - 1

theta = 0.0
theta_history = [theta]
eps = 1e-8
eta = 0.1
while True:
    gradient = dJ(theta)
    last_theta = theta
    theta = theta - eta * gradient
    theta_history.append(theta)
    if (abs(J(theta) - J(last_theta)) < eps):
        break
print(theta)
print(J(theta))
print(theta_history)
plt.plot(plot_x, J(plot_x))
plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
plt.show()
print(len(theta_history))
