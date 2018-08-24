import numpy as np


class Generator:
    def __init__(self, n):
        self.__n = n - 2
        self.elements = []
        self.__containers = [[] for i in range(self.__n)]

    def insert(self, x):
        comb = [[] for i in range(self.__n)]
        comb[0].append(x)

        for c in range(0, self.__n):
            for e in self.__containers[c]:
                if np.sum(x ^ e) == 0:
                    return False
                if c != self.__n - 1:
                    comb[c + 1].append(x ^ e)

        for c in range(len(self.__containers)):
            self.__containers[c] = self.__containers[c] + comb[c]
        self.elements.append(x)
        return True
