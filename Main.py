import numpy as np

from DecoderHamming import DecoderHamming
from EncoderHamming import EncoderHamming
from Channel import Channel


def normal_process(codes, channel):
    output = [channel.add_noise(code) for code in codes]

    return output


def hamming_process(codes, channel):
    # Encoding
    hamming_encoder = EncoderHamming()
    encodes = [hamming_encoder.encode(code) for code in codes]

    # Channeling
    output = [channel.add_noise(code) for code in encodes]

    # Decoding
    hamming_decoder = DecoderHamming()
    output = np.array([hamming_decoder.decode(code) for code in output])

    return output


if __name__ == "__main__":
    # Generating random codes
    codes = []
    n = 1000
    assert (n % 4 == 0)
    for i in range(0, n // 4):
        codes.append(np.rint(np.random.random_sample(4)).astype(bool))

    # Generating outputs
    channel = Channel(0.2)
    # normal_output = normal_process(codes, channel)
    hamming_output = hamming_process(codes, channel)

    # Comparing structures
    hamming_p = np.count_nonzero(np.reshape(hamming_output, (1, n)) == np.reshape(codes, (1, n)))

    print(hamming_p)
