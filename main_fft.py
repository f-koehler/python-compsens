import compsens
import numpy
import matplotlib.pyplot as plt
from scipy import fft

# Size of full signal
n = 1000

# Size of the sample
m = 50

# Time points for full signal
t = numpy.linspace(0, 2 * numpy.pi, n)

# Full signal
y = 0.5 * numpy.cos(2.5 * t) + numpy.cos(3.5 * t)

# Sample indices
samples = numpy.random.choice(n, m, replace=False)
samples.sort()

# Signal samples
s = y[samples]

# Operator to transform solution to temporal domain and perform the sampling.
A = compsens.create_operator_rfft1d(n, samples)

# Compute the solution of the optimization problem
f = compsens.optimize(s, A)

# Reconstruct signal using inverse transformation
y_prime = fft.irfft(f, norm="ortho")


d = t[1] - t[0]
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2)
ax1.plot(t, y)
ax2.plot(fft.rfft(y, norm="ortho").real)
ax3.plot(t[samples], s)
ax5.plot(t, y_prime[:n])
ax6.plot(f.real)

ax1.set_title("original signal")
ax2.set_title("original signal (FFT specrum)")
ax3.set_title("sampled signal")
ax4.set_title("sampled signal (FFT specrum)")
ax5.set_title("reconstructed signal")
ax6.set_title("reconstructed signal (FFT specrum)")

ax2.set_xlim(0, 20)
ax4.set_xlim(0, 20)
ax6.set_xlim(0, 20)

plt.show()
