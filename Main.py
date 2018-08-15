import numpy as np

from DecoderHamming import DecoderHamming
from EncoderHamming import EncoderHamming
from Channel import Channel


def normal_process(codes, channel):
    output = [channel.add_noise(code) for code in codes]

    return output


def hamming_process(codes, channels):
    # Encoding
    hamming_encoder = EncoderHamming()
    encodes = [hamming_encoder.encode(code) for code in codes]

    # Channeling
    outputs = [None] * len(channels)
    for c in range(len(channels)):
        outputs[c] = [channels[c].add_noise(code) for code in encodes]

    # Decoding
    hamming_decoder = DecoderHamming()
    for c in range(len(channels)):
        outputs[c] = np.array([hamming_decoder.decode(code) for code in outputs[c]])

    return outputs


if __name__ == "__main__":
    # Generating random codes
    codes = []
    n = 4
    assert (n % 4 == 0)
    for i in range(0, n // 4):
        codes.append(np.rint(np.random.random_sample(4)).astype(bool))

    # Generating channels
    ps = [0.5, 0.2, 0.1, 5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6]
    channels = [Channel(p) for p in ps]

    # Generating outputs
    # normal_output = normal_process(codes, channels)
    hamming_outputs = hamming_process(codes, channels)

    # Comparing structures
    hamming_ps = []
    for c in range(len(channels)):
        hamming_ps.append(np.count_nonzero(np.reshape(hamming_outputs[c], (1, n)) == np.reshape(codes, (1, n))) / n)

    print(hamming_ps)
