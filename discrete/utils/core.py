from abc import ABC, abstractmethod


class BaseDistributionHandler(ABC):

    @staticmethod
    def quant(prob, n, th, size=1):
        raise NotImplementedError

    @staticmethod
    def cdf(s, n, th, size=1):
        raise NotImplementedError


    @staticmethod
    def pmf(s, n, th, size=1):
        raise NotImplementedError

    @staticmethod
    def rngen(n, th, size=1):
        raise NotImplementedError

    @staticmethod
    def ubound(n, l0, th0, th, size=1):
        raise NotImplementedError
    @staticmethod
    def lbound(n, l1, th1, th, size=1):
        raise NotImplementedError
    @staticmethod
    def hbound(l0, l1, th0, th1, th, size=1):
        raise NotImplementedError
