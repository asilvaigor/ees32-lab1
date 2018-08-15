import numpy as np


class EncoderHamming:
    def encode(self, seq):
        """Encodes a sequence of bytes"""
        encoder_matrix = np.array([[1, 0, 0, 0, 1, 1, 1],
                                   [0, 1, 0, 0, 1, 0, 1],
                                   [0, 0, 1, 0, 1, 1, 0],
                                   [0, 0, 0, 1, 0, 1, 1]])
        return np.mod(np.dot(seq, encoder_matrix), 2)
