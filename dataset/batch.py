import numpy as np

class Batcher(object):
    pass

class WindowedBatcher(object):

    def __init__(self, sequence, encoding, batch_size=100, sequence_length=50):
        self.X = sequence.seq[:, np.newaxis]
        self.encoding = encoding
        self.vocab_size = self.encoding.index
        self.batch_index = 0
        self.batches = []
        self.batch_size = batch_size
        self.sequence_length = sequence_length + 1
        self.length = len(sequence)

        self.batch_index = 0
        N, D = self.X.shape
        assert N > self.batch_size * self.sequence_length, "File has to be at least %u characters" % (self.batch_size * self.sequence_length)

        self.X = self.X[:N - N % (self.batch_size * self.sequence_length)]
        self.N, self.D = self.X.shape
        self.X = self.X.reshape((self.N / self.sequence_length, self.sequence_length, self.D))

        self.N, self.S, self.D = self.X.shape

        self.num_sequences = self.N / self.sequence_length
        self.num_batches = self.N / self.batch_size
        self.batch_cache = {}

    def next_batch(self):
        idx = (self.batch_index * self.batch_size)
        if self.batch_index >= self.num_batches:
            self.batch_index = 0
            idx = 0

        if self.batch_index in self.batch_cache:
            batch = self.batch_cache[self.batch_index]
            self.batch_index += 1
            return batch

        X = self.X[idx:idx + self.batch_size]
        y = np.zeros((X.shape[0], self.sequence_length, self.vocab_size))
        for i in xrange(self.batch_size):
            for c in xrange(self.sequence_length):
                y[i, c, int(X[i, c, 0])] = 1

        assert (y.argmax(axis=2) == X.ravel().reshape(X.shape[:2])).all()
        X = y[:, :-1, :]
        y = y[:, 1:, :]

        assert y.shape[1] == self.sequence_length - 1 and X.shape[1] == self.sequence_length - 1

        X = np.swapaxes(X, 0, 1)
        y = np.swapaxes(y, 0, 1)
        self.batch_cache[self.batch_index] = X, y
        self.batch_index += 1
        return X, y
