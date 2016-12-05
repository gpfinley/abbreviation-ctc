# Read in a .bin file as saved in word2vec, which stores words in text but vectors in binary format

__author__ = 'gpfinley'

import struct
import time
import numpy as np
import logging

def get_dictionary(binfile):

    f = open(binfile)
    content = f.read()
    f.close()

    # Header of the file contains two integers: the number of words and the vector dimensionality
    header = content[:content.find('\n')]
    words, size = int(header.split()[0]), int(header.split()[1])
    rawvecs = content[content.find('\n')+1:]

    vectors = {}
    print header

    start = time.time()

    # Move through the rest of it character by character
    word = ''
    i = 0
    while True:
        if i >= len(rawvecs):
            break
        c = rawvecs[i]
        if c == ' ':
            # Move the index past the space and unpack the vector, four bytes at a time
            i += 1
            vector = []
            for x in range(size):
                vector.append(struct.unpack('f', rawvecs[x*4+i:(x+1)*4+i])[0])
            # Set i to the beginning of the next line
            i += size*4 + 1
            vectors[word] = np.array(vector)
            word = ''
        else:
            word += c
            i += 1

    logging.info('Read', len(vectors), 'words in', time.time() - start, 'seconds')
    return vectors
