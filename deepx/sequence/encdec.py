import numpy as np
import theano.tensor as T
from theanify import theanify

import logging

from deepx.nn import LSTM

class Encoder(LSTM):

    def __init__(self, name, n_input, n_hidden, max_seq_length,
                 num_layers=2,
                 use_forget_gate=True,
                 use_input_peep=False, use_output_peep=False, use_forget_peep=False,
                 use_tanh_output=True, seed=None):
        super(Encoder, self).__init__(name, n_input, n_hidden,
                                      num_layers=num_layers,
                                      use_forget_gate=use_forget_gate,
                                      use_input_peep=use_input_peep, use_output_peep=use_output_peep,
                                      use_forget_peep=use_forget_peep,
                                      use_tanh_output=use_tanh_output,
                                      seed=seed)
        self.max_seq_length = max_seq_length

    @theanify(T.tensor3('X'), T.matrix('mask'), T.tensor3('previous_hidden'), T.tensor3('previous_state'))
    def encode(self, X, mask, previous_hidden, previous_state):
        """
        Parameters:
            X - (n_batch, max_seq_length, n_input) tensor3
            mask - (n_batch, max_seq_length) matrix (binary)
            previous_hidden - (n_batch, num_layers, n_hidden) tensor3
            previous_state - (n_batch, num_layers, n_hidden) tensor3

        Returns:
            hidden - (n_batch, n_layers, n_hidden)
            state - (n_batch, n_layers, n_hidden)
        """
        for i in xrange(self.max_seq_length):
            candidate_hidden, candidate_state = self.step(X[:, i, :], previous_hidden, previous_state)
            mask_i = mask[:, i, np.newaxis, np.newaxis]
            previous_hidden = candidate_hidden * mask_i + previous_hidden * (1 - mask_i)
            previous_state = candidate_state * mask_i + previous_state * (1 - mask_i)
        return previous_hidden, previous_state

class Decoder(LSTM):

    def __init__(self, name, n_input, n_hidden, max_seq_length, output, word_matrix,
                 num_layers=2,
                 use_forget_gate=True,
                 use_input_peep=False, use_output_peep=False, use_forget_peep=False,
                 use_tanh_output=True, seed=None):
        super(Decoder, self).__init__(name, n_input, n_hidden,
                                      num_layers=num_layers,
                                      use_forget_gate=use_forget_gate,
                                      use_input_peep=use_input_peep, use_output_peep=use_output_peep,
                                      use_forget_peep=use_forget_peep,
                                      use_tanh_output=use_tanh_output,
                                      seed=seed)
        self.max_seq_length = max_seq_length
        self.output = output
        self.word_matrix = word_matrix

    @theanify(T.matrix('X'), T.matrix('mask'), T.tensor3('previous_hidden'), T.tensor3('previous_state'))
    def decode(self, X, mask, previous_hidden, previous_state):
        """
        Parameters:
            X - (n_batch, n_input) matrix
            mask - (n_batch, max_seq_length) matrix (binary)
            previous_hidden - (n_batch, num_layers, n_hidden) tensor3
            previous_state - (n_batch, num_layers, n_hidden) tensor3

        Returns:
            outputs - (max_seq_length, n_batch, vocab_size)
            hiddens - (max_seq_length, n_batch, n_layers, n_hidden)
            states - (max_seq_length, n_batch, n_layers, n_hidden)
        """
        outputs = []
        input = X
        for i in xrange(self.max_seq_length):
            candidate_hidden, candidate_state = self.step(input, previous_hidden, previous_state)
            mask_i = mask[:, i, np.newaxis, np.newaxis]
            previous_hidden = candidate_hidden * mask_i + previous_hidden * (1 - mask_i)
            previous_state = candidate_state * mask_i + previous_state * (1 - mask_i)
            output = self.output.forward(candidate_hidden[:, -1, :])
            outputs.append(output)
            input = self.word_matrix[output.argmax(axis=1)]
        return T.swapaxes(T.stack(*outputs), 0, 1), previous_hidden, previous_state

    def get_parameters(self):
        return super(Decoder, self).get_parameters() + self.output.get_parameters()

if __name__ == "__main__":
    num_layers = 2
    n_input = 4
    n_hidden = 10
    n_batch = 15
    max_seq_length = 20
    lstm = Encoder('encoder', n_input, n_hidden, max_seq_length,
                   num_layers=num_layers, seed=1).compile()
    X = np.ones((n_batch, max_seq_length, n_input))
    sequence_length = range(6, 21)
    mask = np.zeros((n_batch, max_seq_length))
    for i in xrange(n_batch):
        for j in xrange(sequence_length[i]):
            mask[i, j] = 1

    previous_hidden = np.zeros((n_batch, num_layers, n_hidden))
    previous_state = np.zeros((n_batch, num_layers, n_hidden))
    h, s = lstm.encode(X, mask, previous_hidden, previous_state)
