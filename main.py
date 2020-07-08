import compsens
import numpy
import matplotlib.pyplot as plt
from scipy import fftpack as fft

# Size of full signal
n = 1000

# Size of the sample
m = 50

# Time points for full signal
t = numpy.linspace(0, 2 * numpy.pi, n)

# Full signal
y = 0.5 * numpy.cos(2.5 * t) + numpy.cos(5.4 * t) - numpy.cos(3.9 * t + 2)

# Sample indices
samples = numpy.random.choice(n, m, replace=False)
samples.sort()

# Signal samples
s = y[samples]

# Operator to transform solution to temporal domain and perform the sampling.
A = compsens.create_operator_dct1d(n, samples)

# Compute the solution of the optimization problem
f = compsens.optimize(s, A)

# Reconstruct signal using inverse transformation
y_prime = fft.idct(f, norm="ortho", axis=0)

fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
ax1.plot(t, y)
ax2.plot(fft.dct(y, norm="ortho", axis=0))
ax3.plot(t[samples], s)
ax4.plot(fft.dct(s, norm="ortho", axis=0))
ax5.plot(t, y_prime)
ax6.plot(f)

ax2.set_xlim(0, 20)
ax4.set_xlim(0, 20)
ax6.set_xlim(0, 20)

plt.show()
