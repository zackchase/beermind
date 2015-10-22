import numpy as np
import theano
import theano.tensor as T

from theanify import theanify
from deepx.nn import ParameterModel, LSTM, Softmax

class CharacterRNN(ParameterModel):

    def __init__(self, name, encoding, n_hidden=10, n_layers=2):
        super(CharacterRNN, self).__init__(name)
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.vocab_size = encoding.index

        self.lstm = LSTM('%s-charrnn' % name, self.vocab_size,
                         n_hidden=self.n_hidden,
                         n_layers=self.n_layers)
        self.output = Softmax('%s-softmax' % name, n_hidden, self.vocab_size)

    @theanify(T.tensor3('X'), T.tensor3('state'), T.tensor3('y'))
    def cost(self, X, state, y):
        _, state, ypred = self.forward(X, state)
        S, N, V = y.shape
        y = y.reshape((S * N, V))
        ypred = ypred.reshape((S * N, V))
        return T.nnet.categorical_crossentropy(ypred, y).mean(), state

    @theanify(T.tensor3('X'), T.tensor3('state'))
    def forward(self, X, state):
        S, N, D = X.shape
        H = self.lstm.n_hidden
        L = self.lstm.n_layers
        O = self.output.n_output

        def step(input, previous_hidden, previous_state, previous_output):
            lstm_hidden, state = self.lstm.forward(input, previous_hidden, previous_state)
            final_output = self.output.forward(lstm_hidden[:, -1, :])
            return lstm_hidden, state, final_output

        hidden = T.unbroadcast(T.alloc(np.array(0).astype(theano.config.floatX), N, L, H), 1)

        (encoder_output, encoder_state, softmax_output), _ = theano.scan(step,
                              sequences=[X],
                              outputs_info=[
                                            hidden,
                                            state,
                                            T.alloc(np.asarray(0).astype(theano.config.floatX),
                                                    N,
                                                    O),
                                           ],
                              n_steps=S)
        return encoder_output, encoder_state, softmax_output

    @theanify(T.vector('start_token'), T.scalar('length'), T.scalar('temperature'))
    def generate(self, start_token, length, temperature):
        N = 1
        H = self.lstm.n_hidden
        L = self.lstm.n_layers
        O = self.output.n_output

        def step(input, previous_hidden, previous_state):
            lstm_hidden, state = self.lstm.forward(input, previous_hidden, previous_state)
            final_output = self.output.forward(lstm_hidden[:, -1, :])
            return final_output, lstm_hidden, state

        hidden = T.unbroadcast(T.alloc(np.array(0).astype(theano.config.floatX), N, L, H), 1)
        state = T.unbroadcast(T.alloc(np.array(0).astype(theano.config.floatX), N, L, H), 1)

        (encoder_output, encoder_state, softmax_output), _ = theano.scan(step,
                              outputs_info=[
                                            start_token,
                                            hidden,
                                            state,
                                            T.alloc(np.asarray(0).astype(theano.config.floatX),
                                                    N,
                                                    O),
                                           ],
                              n_steps=length)
        return encoder_output, encoder_state, softmax_output

    def get_parameters(self):
        return self.lstm.get_parameters() + self.output.get_parameters()
