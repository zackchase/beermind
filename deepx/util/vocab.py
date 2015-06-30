import numpy as np

class Vocabulary(object):

    def __init__(self, w2v, word_size):
        self.w2v = w2v
        self.word_size = word_size
        self.forward_map = {}
        self.backward_map = {}
        self.vocab_size = 0

        for word in self.w2v.vocab:
            self.forward_map[word] = self.vocab_size
            self.backward_map[self.vocab_size] = word
            self.vocab_size += 1

        self.word_matrix = np.zeros((self.vocab_size, self.word_size))

        for i in xrange(self.vocab_size):
            self.word_matrix[i] = self.w2v[self.backward_map[i]]

    def get_vector(self, word):
        return self.w2v[word]
