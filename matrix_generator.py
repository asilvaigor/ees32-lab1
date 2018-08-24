import numpy as np
import itertools as it

from Generator import Generator


def kbits(n, k):
    """ Generates a list of combinations of n bits with k of them being 1. """

    result = []
    for bits in it.combinations(range(n), k):
        s = ['0'] * n
        for bit in bits:
            s[bit] = '1'
        result.append(''.join(s))
    return result


if __name__ == "__main__":
    dist = 5
    n = 9
    size = 21

    # GENERATING INITIAL ROWS (IDENTITY)
    generator = Generator(dist)
    id = np.identity(n, dtype=int)
    for row in id:
        generator.insert(row)

    # GENERATING OTHER ROWS POSSIBILITIES
    arr = []
    for i in range(dist - 1, n + 1):
        arr += [list(el) for el in kbits(n, i)]

    # INSERTING MORE ROWS IF POSSIBLE
    elements = np.array(arr, dtype=int)
    for e in range(len(elements)):
        generator.insert(elements[e])

    # VALIDATION
    elements = np.array(generator.elements[0:size])
    print(elements)
    print("Number of rows of Ht: ", len(generator.elements))

    for s in range(1, dist):
        errors = []
        for i in range(1, s + 1):
            errors += [np.array(list(el), dtype=int) for el in kbits(size, i)]
        sindromes = np.mod([np.dot(error, elements) for error in errors], 2)
        print("Number of sindromes using at most " + str(s) + " bits of error: ", len(sindromes))
        print("Number of unique sindromes using at most " + str(s) + " bits of error: ",
              len(set([str(s) for s in sindromes])))
