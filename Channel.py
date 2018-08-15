from random import random as rand
import numpy


class Channel:

    def __init__(self, p):
        self.__p = p

    def get_p(self):
        return self.__p

    def set_p(self, p):
        self.__p = p

    def add_noise(self, v):
        """
        Receives a list of bits v and return a new list representing v with noise applied
        :param v: list of bits
        :return: new list
        """
        idx = 0
        for bit in v:
            num = rand()
            bit = int(bit)
            if num < self.get_p():
                bit = (bit+1) % 2
            bit = bool(bit)
            v[idx] = bit
            idx += 1
        return v
