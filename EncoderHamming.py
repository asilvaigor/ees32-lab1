import numpy as np

class EncoderHamming:
    def encode(self, seq):
        """Encodes a sequence of bytes"""
        encoder_matrix = np.matrix([[1, 0, 0, 0, 1, 1, 1],
                                    [0, 1, 0, 0, 1, 0, 1],
                                    [0, 0, 1, 0, 1, 1, 0],
                                    [0, 0, 0, 1, 0, 1, 1]], dtype=bool)
        return np.matmul(encoder_matrix, seq)
