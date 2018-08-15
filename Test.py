import numpy as np
import time
from .EncoderHamming import EncoderHamming
from .DecoderHamming import DecoderHamming
from .Channel import Channel


if __name__ == 'main':
    # Instantiating encoder, decoder and channel
    encoder = EncoderHamming()
    decoder = DecoderHamming()
    channel = Channel()

    # Receiving code
    code = np.matrix([[bool(x) for x in input().split()]])

    # Printing code
    print(code)
    time.sleep(3)

    # Encoding channel
