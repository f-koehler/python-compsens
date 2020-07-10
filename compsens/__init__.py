from typing import Union

from scipy import fft
import numpy
import cvxpy


def get_random_sample_indices_1d(
    signal: numpy.ndarray, samples: Union[int, float]
) -> numpy.ndarray:
    n = len(signal)

    if isinstance(samples, float):
        m = int(numpy.round(len(signal) * samples))
    elif isinstance(samples, int):
        m = samples
    else:
        raise TypeError("samples should be either a float or an int")

    indices = numpy.random.choice(n, m, replace=False)
    indices.sort()
    return indices


def get_sample_indices_1d(signal: numpy.ndarray, stride: int):
    n = len(signal)
    return numpy.arange(0, n, stride)


def optimize(s: numpy.ndarray, A: numpy.ndarray):
    m = A.shape[0]
    n = A.shape[1]

    assert len(s.shape) == 1
    assert s.shape[0] == m

    f = cvxpy.Variable(n)
    objective = cvxpy.Minimize(cvxpy.norm(f, 1))
    constraints = [
        A @ f == s,
    ]
    problem = cvxpy.Problem(objective, constraints)
    problem.solve(verbose=True)
    return numpy.array(f.value)
