import numpy
from scipy import fft
import cvxpy


def create_operator_dct1d(n: int, samples: numpy.ndarray) -> numpy.ndarray:
    return fft.idct(numpy.identity(n), norm="ortho", axis=0)[samples]


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
