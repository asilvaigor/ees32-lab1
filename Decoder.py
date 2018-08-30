import itertools as it
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
        nrows = self.__ht.shape[0]
        ncols = self.__ht.shape[1]

        map = np.zeros((pow(2, ncols), nrows), dtype=bool)

        # Mapping for no errors
        map[0] = np.zeros(nrows)

        # Mapping for errors of 1 bit
        errors1 = [np.array(list(el), dtype=int) for el in self.__kbits(nrows, 1)]
        indexes1 = [self.__bin2int(np.mod(np.dot(error, self.__ht), 2)) for error in errors1]
        for i in range(len(indexes1)):
            if (np.count_nonzero(map[indexes1[i]])) == 0:
                map[indexes1[i]] = errors1[i]

        # Mapping for errors of 2 bits, prioritizing errors of 1 bit
        errors2 = [np.array(list(el), dtype=int) for el in self.__kbits(nrows, 2)]
        indexes2 = [self.__bin2int(np.mod(np.dot(error, self.__ht), 2)) for error in errors2]
        for i in range(len(indexes2)):
            if np.count_nonzero(map[indexes2[i]]) == 0:
                map[indexes2[i]] = errors2[i]

        # Mapping for errors of 3 bits, prioritizing errors of 2 and 1 bits
        errors3 = [np.array(list(el), dtype=int) for el in self.__kbits(nrows, 3)]
        indexes3 = [self.__bin2int(np.mod(np.dot(error, self.__ht), 2)) for error in errors3]
        for i in range(len(indexes3)):
            if np.count_nonzero(map[indexes3[i]]) == 0:
                map[indexes3[i]] = errors3[i]

        return map

    @staticmethod
    def __kbits(n, k):
        """ Generates a list of combinations of n bits with k of them being 1. """

        result = []
        for bits in it.combinations(range(n), k):
            s = ['0'] * n
            for bit in bits:
                s[bit] = '1'
            result.append(''.join(s))
        return result

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

        return (code ^ error)[0:int(self.__ht.shape[0] * 4 / 7)]
