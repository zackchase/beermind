import numpy as np
import theano
import theano.tensor as T

from theanify import theanify
from deepx.nn import ParameterModel, LSTMLayer, Softmax

class LSTM(ParameterModel):
    def __init__(self, name, n_input, n_hidden=10, n_layers=2):
        super(LSTM, self).__init__(name)

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        assert self.n_layers >= 1
        self.layers = []
        self.input_layer = LSTMLayer('%s-input' % name,
                                     self.n_input,
                                     self.n_hidden)
        for i in xrange(self.n_layers - 1):
            self.layers.append(LSTMLayer('%s-layer-%u' % (name, i),
                                         self.n_hidden,
                                         self.n_hidden))

    def forward(self, X, previous_state, previous_hidden):
        output, state = self.input_layer.step(X, previous_state[:, 0, :], previous_hidden[:, 0, :])
        hiddens, states = [output], [state]
        for i, layer in enumerate(self.layers):
            output, state = layer.step(output, previous_state[:, i + 1, :], previous_hidden[:, i + 1, :])
            hiddens.append(output)
            states.append(state)
        return T.swapaxes(T.stack(*hiddens), 0, 1), T.swapaxes(T.stack(*states), 0, 1)

    def get_parameters(self):
        params = self.input_layer.get_parameters()
        for layer in self.layers:
            params += layer.get_parameters()
        return params

class CharacterRNN(ParameterModel):

    def __init__(self, name, encoding, n_hidden=10, n_layers=2):
        super(CharacterRNN, self).__init__(name)
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.vocab_size = encoding.index

        self.lstm = LSTM('%s-charrnn' % name, 1,
                         n_hidden=self.n_hidden,
                         n_layers=self.n_layers)
        self.output = Softmax('%s-softmax' % name, n_hidden, self.vocab_size)

    @theanify(T.tensor3('X'), T.tensor3('state'), T.tensor3('y'))
    def cost(self, X, state, y):
        _, state, ypred = self.forward(X, state)
        S, N, V = y.shape
        y = y.reshape((S * N, V))
        ypred = ypred.reshape((S * N, V))
        return T.nnet.categorical_crossentropy(ypred, y).sum(), state

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

    def get_parameters(self):
        return self.lstm.get_parameters() + self.output.get_parameters()
