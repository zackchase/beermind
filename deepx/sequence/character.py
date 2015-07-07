import numpy as np
import theano
import theano.tensort as T

from theanify import theanify
from deepx.nn import ParameterModel, LSTM, Softmax

class CharacterGenerator(ParameterModel):

    def __init__(self, name, vocab, n_hidden, num_layers=2):
        super(CharacterGenerator, self).__init__(name)
        self.vocab = vocab
        self.n_hidden = n_hidden
        self.num_layers = num_layers

        self.vocab_size = vocab.vocab_size

        self.lstm = LSTM('encoder', 1, self.n_hidden, num_layers=self.num_layers)
        self.softmax = Softmax('softmax', n_hidden, self.vocab_size)

        self.average_gradient = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]
        self.average_rms = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]
        self.parameter_update = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]

    @theanify(T.tensor3('X'), T.tensor3('state'))
    def forward(self, X, state):
        S, N, D = X.shape
        H = self.lstm.n_hidden
        L = self.lstm.num_layers
        O = self.output.n_output

        def step(input, previous_hidden, previous_state, previous_output):
            lstm_hidden, state = self.lstm.step(input, previous_hidden, previous_state)
            final_output = self.output.forward(lstm_hidden[:, -1, :])
            return lstm_hidden, state, final_output

        (encoder_output, encoder_state, softmax_output), _ = theano.scan(step,
                              sequences=[X],
                              outputs_info=[T.alloc(np.asarray(0).astype(theano.config.floatX),
                                                    N,
                                                    L,
                                                    H),
                                            state,
                                            T.alloc(np.asarray(0).astype(theano.config.floatX),
                                                    N,
                                                    O),
                                           ],
                              n_steps=S)
        return encoder_output, encoder_state, softmax_output
