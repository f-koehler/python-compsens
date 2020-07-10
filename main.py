import numpy
import matplotlib.pyplot as plt

import compsens
from compsens.trafo import DST1D

# Size of full signal
n = 1000

# Size of the sample
m = 100

# Time points for full signal
t = numpy.linspace(0, 6 * numpy.pi, n)

# Full signal
y = 0.5 * numpy.sin(2.5 * t) + numpy.sin(3.5 * t)

# Sample indices
samples = compsens.get_random_sample_indices_1d(y, 0.1)

# Signal samples
s = y[samples]

# Operator to transform solution to temporal domain and perform the sampling.
trafo = DST1D()

# Compute the solution of the optimization problem
f = compsens.optimize(s, trafo.get_sampling_matrix(n, samples))

# Reconstruct signal using inverse transformation
y_prime = trafo.transform_back(f)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.plot(t, y)
ax1.plot(t[samples], s, ".")
ax2.plot(trafo.transform(y))
ax3.plot(t, y_prime)
ax4.plot(f)

ax1.set_title("original signal")
ax2.set_title("original signal (DCT)")
ax3.set_title("reconstructed signal")
ax4.set_title("reconstructed signal (DCT)")

ax2.set_xlim(0, 100)
ax4.set_xlim(0, 100)

plt.show()
