import numpy as np
import time
from EncoderHamming import EncoderHamming
from DecoderHamming import DecoderHamming
from Channel import Channel


if __name__ == '__main__':
    # Instantiating encoder, decoder and channel
    encoder = EncoderHamming()
    decoder = DecoderHamming()
    channel = Channel(0.3)

    # Receiving code
    code = np.matrix([[int(x) for x in input()]], dtype=bool)

    # Printing code
    print("Code received: ")
    print(code)
    time.sleep(1)

    # Encoding channel
    print("Encoding: ")
    encoded = encoder.encode(code)
    print(encoded)
    time.sleep(1)

    # Passing through channel
    print("Passing through channel: ")
    # [int(x) for x in encoded]
    # through_channel = channel.add_noise()
    # print(through_channel)
    # time.sleep(4)

    # Decoding code
    print("Decoding: ")
    decoded = decoder.decode(encoded)
    print(decoded)