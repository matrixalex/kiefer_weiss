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
