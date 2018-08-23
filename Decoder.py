import numpy as np


class Decoder:
    def __init__(self, P):
        self.__ht = self.__concatenate(P)
        self.tot = sum(P.shape)
        self.__map = self.__get_sindrome_map()

    def setParity(self, P):
        self.__ht = self.__concatenate(P)
        self.tot = sum(P.shape)

    @staticmethod
    def __concatenate(P):
        """Concatenates identity matrix in the beginning of P"""
        n = P.shape[1]
        return np.concatenate([P, np.identity(n, dtype=int)], 0)

    def __get_sindrome_map(self):
        map = np.zeros((self.__ht.shape[0] + 1, self.__ht.shape[0]), dtype=bool)
        map[0, :] = np.array((0, 0, 0, 0, 0, 0, 0), dtype=bool)
        map[1, :] = np.array((0, 0, 0, 0, 0, 0, 1), dtype=bool)
        map[2, :] = np.array((0, 0, 0, 0, 0, 1, 0), dtype=bool)
        map[3, :] = np.array((0, 0, 0, 1, 0, 0, 0), dtype=bool)
        map[4, :] = np.array((0, 0, 0, 0, 1, 0, 0), dtype=bool)
        map[5, :] = np.array((0, 1, 0, 0, 0, 0, 0), dtype=bool)
        map[6, :] = np.array((0, 0, 1, 0, 0, 0, 0), dtype=bool)
        map[7, :] = np.array((1, 0, 0, 0, 0, 0, 0), dtype=bool)
        return map

    @staticmethod
    def __bin2int(x):
        y = 0
        for i, j in enumerate(x):
            y += j << i
        return y

    def decode(self, code):
        assert (code.shape == (self.__ht.shape[0],))
        sindrome = np.mod(np.dot(code, self.__ht), 2)
        error = self.__map[self.__bin2int(sindrome)]

        return (code ^ error)[0:4]
