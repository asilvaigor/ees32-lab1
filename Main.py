# import matplotlib as plt
import numpy as np
import time

from DecoderHamming import DecoderHamming
from EncoderHamming import EncoderHamming
from Channel import Channel

# Script which generates N random bits and simulates a random channel with probabilities ranging from 0.5 to 10e-6.
# It then plots a graph comparing different encoding processes.
N = 10000


def normal_process(codes, channels):
    outputs = [None] * len(channels)
    for c in range(len(channels)):
        outputs[c] = np.array([channels[c].add_noise(code) for code in codes])

    return outputs


def hamming_process(codes, channels):
    # Encoding
    hamming_encoder = EncoderHamming()
    encodes = [hamming_encoder.encode(code) for code in codes]

    # Channeling
    outputs = [None] * len(channels)
    for c in range(len(channels)):
        outputs[c] = np.array([channels[c].add_noise(code) for code in encodes])

    # Decoding
    hamming_decoder = DecoderHamming()
    for c in range(len(channels)):
        outputs[c] = np.array([hamming_decoder.decode(code) for code in outputs[c]])

    return outputs


if __name__ == "__main__":
    t = time.time()

    # Generating random codes
    codes = []
    assert (N % 4 == 0)
    for i in range(0, N // 4):
        codes.append(np.rint(np.random.random_sample(4)).astype(bool))

    # Generating channels with different noises to plot a graph
    ps = [0.5, 0.2, 0.1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6]
    channels = [Channel(p) for p in ps]

    # Generating outputs without encoding, with hamming encoding and with our encoding
    normal_outputs = normal_process(codes, channels)
    hamming_outputs = hamming_process(codes, channels)

    # Comparing outputs and plotting a graph
    normal_ps = []
    hamming_ps = []
    for c in range(len(channels)):
        normal_ps.append(1 - np.count_nonzero(np.reshape(normal_outputs[c], (1, N)) == np.reshape(codes, (1, N))) / N)
        hamming_ps.append(1 - np.count_nonzero(np.reshape(hamming_outputs[c], (1, N)) == np.reshape(codes, (1, N))) / N)

    np.set_printoptions(linewidth=120)
    print("Time taken:", time.time() - t, "s")
    print("Probabilities:", ps)
    print("No encoding:", normal_ps)
    print("Hamming encoding:", hamming_ps)
