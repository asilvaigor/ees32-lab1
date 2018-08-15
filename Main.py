import numpy as np

from DecoderHamming import DecoderHamming
from EncoderHamming import EncoderHamming
#from Channel import Channel

if __name__ == "__main__":
    # Generating random codes
    codes = []
    for i in range(0, 250000):
        codes.append(np.rint(np.random.random_sample(7)).astype(bool))

    # Encoding
    hamming_encoder = EncoderHamming()
    hamming_encodes = [hamming_encoder.encode(code) for code in codes]

    # Channeling
    channel = Channel()
    hamming_output = None

    # Decoding
    hamming_decoder = DecoderHamming()
    hamming_decodes = [hamming_decoder.decode(code) for code in hamming_output]

    # Comparing structures
