from abc import abstractmethod
from typing import Optional

import numpy
from scipy import fft


class Trafo1D:
    @abstractmethod
    def transform(self, signal: numpy.ndarray) -> numpy.ndarray:
        del self
        del signal

    @abstractmethod
    def transform_back(self, signal: numpy.ndarray) -> numpy.ndarray:
        del self
        del signal

    @abstractmethod
    def get_sampling_matrix(self, n: int, samples: numpy.ndarray) -> numpy.ndarray:
        del self
        del n
        del samples


class DCT1D(Trafo1D):
    def __init__(self, type: int = 2, norm: Optional[str] = "ortho"):
        self.type = type
        self.norm = norm

    def transform(self, signal: numpy.ndarray) -> numpy.ndarray:
        return fft.dct(signal, type=self.type, norm=self.norm)

    def transform_back(self, signal: numpy.ndarray) -> numpy.ndarray:
        return fft.idct(signal, type=self.type, norm=self.norm)

    def get_sampling_matrix(self, n: int, samples: numpy.ndarray) -> numpy.ndarray:
        return fft.idct(numpy.identity(n), type=self.type, norm=self.norm, axis=0)[
            samples
        ]


class DST1D(Trafo1D):
    def __init__(self, type: int = 2, norm: Optional[str] = "ortho"):
        self.type = type
        self.norm = norm

    def transform(self, signal: numpy.ndarray) -> numpy.ndarray:
        return fft.dst(signal, type=self.type, norm=self.norm)

    def transform_back(self, signal: numpy.ndarray) -> numpy.ndarray:
        return fft.idst(signal, type=self.type, norm=self.norm)

    def get_sampling_matrix(self, n: int, samples: numpy.ndarray) -> numpy.ndarray:
        return fft.idst(numpy.identity(n), type=self.type, norm=self.norm, axis=0)[
            samples
        ]


class RFFT1D(Trafo1D):
    def __init__(self, norm: Optional[str] = "ortho"):
        self.norm = norm

    def transform(self, signal: numpy.ndarray) -> numpy.ndarray:
        return fft.rfft(signal, norm=self.norm)

    def transform_back(self, signal: numpy.ndarray) -> numpy.ndarray:
        return fft.irfft(signal, norm=self.norm)

    def get_sampling_matrix(self, n: int, samples: numpy.ndarray) -> numpy.ndarray:
        return fft.irfft(numpy.identity(n), norm=self.norm, axis=0)[samples]
