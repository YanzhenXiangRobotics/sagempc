import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the parametric function
def parametric_curve(t, a, b, c, d):
    x = a * np.sin(b * t + c)
    y = d * np.cos(b * t + c)
    return x, y

# Sample data points
t_data = np.linspace(0, 2 * np.pi, 100)
# x_data = 2 * np.sin(1.5 * t_data + 0.5) + np.random.normal(size=t_data.size)
# y_data = 3 * np.cos(1.5 * t_data + 0.5) + np.random.normal(size=t_data.size)
x_data = 2 * np.sin(1.5 * t_data + 0.5)
y_data = 3 * np.cos(1.5 * t_data + 0.5)

# Fit the curve
params, _ = curve_fit(lambda t, a, b, c, d: parametric_curve(t, a, b, c, d)[0], t_data, x_data)
params_y, _ = curve_fit(lambda t, a, b, c, d: parametric_curve(t, a, b, c, d)[1], t_data, y_data)

# Plot the results
plt.figure()
plt.scatter(x_data, y_data, label='Data')
t_fit = np.linspace(0, 2 * np.pi, 1000)
x_fit, y_fit = parametric_curve(t_fit, *params)
plt.plot(x_fit, y_fit, label='Fitted curve', color='red')
plt.legend()
plt.show()
