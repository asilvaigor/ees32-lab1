import numpy as np


class Decoder:
    def __init__(self, P):
        self.__ht = self.__concatenate(P)
        self.tot = sum(P.shape)

    def setParity(self, P):
        self.__ht = self.__concatenate(P)
        self.tot = sum(P.shape)

    @staticmethod
    def __concatenate(P):
        """Concatenates identity matrix in the beginning of P"""
        n = P.shape[1]
        return np.concatenate([P, np.identity(n, dtype=int)], 0)

    def decode(self, code):
        assert(code.shape == (7,))
        sindrome = np.mod(np.dot(code, self.__ht), 2)
        error = np.zeros(self.tot, dtype=bool)

        for i in range(self.tot):


        return (code ^ error)[0:4]
