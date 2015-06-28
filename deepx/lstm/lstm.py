import numpy as np
import theano
import theano.tensor as T

from theanify import theanify, Theanifiable

class LSTM(Theanifiable):

    def __init__(self, name, n_input, n_hidden, num_layers=2, use_forget_gate=True):
        super(LSTM, self).__init__()
        self.name = name
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.use_forget_gate = use_forget_gate

        assert self.num_layers >= 1

        self.parameters = {}

        self.set_parameter('Wi', (np.random.rand(self.n_input, self.n_hidden) - 0.5) / 1000.0)
        self.set_parameter('Ui', (np.random.rand(self.num_layers, self.n_hidden, self.n_hidden) - 0.5) / 1000.0)
        self.set_parameter('bi', (np.random.rand(self.num_layers, self.n_hidden) - 0.5) / 1000.0)

        self.set_parameter('Whi', (np.random.rand(self.num_layers - 1, self.n_hidden, self.n_hidden) - 0.5) / 1000.0)

        self.set_parameter('Wf', (np.random.rand(self.n_input, self.n_hidden) - 0.5) / 1000.0)
        self.set_parameter('Uf', (np.random.rand(self.num_layers, self.n_hidden, self.n_hidden) - 0.5) / 1000.0)
        self.set_parameter('bf', (np.random.rand(self.num_layers, self.n_hidden) - 0.5) / 1000.0)

        self.set_parameter('Whf', (np.random.rand(self.num_layers - 1, self.n_hidden, self.n_hidden) - 0.5) / 1000.0)

        self.set_parameter('Wc', (np.random.rand(self.n_input, self.n_hidden) - 0.5) / 1000.0)
        self.set_parameter('Uc', (np.random.rand(self.num_layers, self.n_hidden, self.n_hidden) - 0.5) / 1000.0)
        self.set_parameter('bc', (np.random.rand(self.num_layers, self.n_hidden) - 0.5) / 1000.0)

        self.set_parameter('Whc', (np.random.rand(self.num_layers - 1, self.n_hidden, self.n_hidden) - 0.5) / 1000.0)

        self.set_parameter('Wo', (np.random.rand(self.n_input, self.n_hidden) - 0.5) / 1000.0)
        self.set_parameter('Vo', (np.random.rand(self.num_layers, self.n_hidden, self.n_hidden) - 0.5) / 1000.0)
        self.set_parameter('Uo', (np.random.rand(self.num_layers, self.n_hidden, self.n_hidden) - 0.5) / 1000.0)
        self.set_parameter('bo', (np.random.rand(self.num_layers, self.n_hidden) - 0.5) / 1000.0)

        self.set_parameter('Who', (np.random.rand(self.num_layers - 1, self.n_hidden, self.n_hidden) - 0.5) / 1000.0)

    def set_parameter(self, name, value):
        self.parameters[name] = theano.shared(value, name='%s-%s' % (self.name, name))

    def get_parameter(self, name):
        return self.parameters[name]

    def set_parameter_value(self, name, value):
        return self.parameters[name].set_value(value)

    def get_parameter_value(self, name):
        return self.parameters[name].get_value()

    def forward_with_weights(self, X, previous_hidden, previous_state, Wi, Ui, bi, Wf, Uf, bf, Wc, Uc, bc, Wo, Vo, Uo, bo):
        input_gate      = T.nnet.sigmoid(T.dot(X, Wi) + T.dot(previous_hidden, Ui) + bi)
        candidate_state = T.tanh(T.dot(X, Wc) + T.dot(previous_hidden, Uc) + bc)

        if self.use_forget_gate:
            forget_gate     = T.nnet.sigmoid(T.dot(X, Wf) + T.dot(previous_hidden, Uf) + bf)
            state           = candidate_state * input_gate + previous_state * forget_gate
        else:
            state           = candidate_state * input_gate + previous_state * 0

        output_gate     = T.nnet.sigmoid(T.dot(X, Wo) + T.dot(previous_hidden, Uo) \
                                        + T.dot(state, Vo) + bo)
        output          = output_gate * T.tanh(state)
        return output, state

    @theanify(T.matrix('X'), T.tensor3('previous_hidden'), T.tensor3('previous_state'))
    def step(self, X, previous_hidden, previous_state):
        out, state = self.forward_with_weights(X, previous_hidden[:, 0, :], previous_state[:, 0, :],
                                               self.get_parameter('Wi'), self.get_parameter('Ui')[0], self.get_parameter('bi')[0],
                                               self.get_parameter('Wf'), self.get_parameter('Uf')[0], self.get_parameter('bf')[0],
                                               self.get_parameter('Wc'), self.get_parameter('Uc')[0], self.get_parameter('bc')[0],
                                               self.get_parameter('Wo'), self.get_parameter('Vo')[0], self.get_parameter('Uo')[0], self.get_parameter('bo')[0])
        outs = [out]
        states = [state]
        for l in xrange(1, self.num_layers):
            out, state = self.forward_with_weights(out, previous_hidden[:, l, :], previous_state[:, l, :],
                                                self.get_parameter('Whi')[l - 1], self.get_parameter('Ui')[l], self.get_parameter('bi')[l],
                                                self.get_parameter('Whf')[l - 1], self.get_parameter('Uf')[l], self.get_parameter('bf')[l],
                                                self.get_parameter('Whc')[l - 1], self.get_parameter('Uc')[l], self.get_parameter('bc')[l],
                                                self.get_parameter('Who')[l - 1], self.get_parameter('Vo')[l], self.get_parameter('Uo')[l], self.get_parameter('bo')[l])
            states.append(state)
            outs.append(out)
        return T.swapaxes(T.stack(*outs), 0, 1), T.swapaxes(T.stack(*states), 0, 1)

    def get_parameters(self):
        params = self.parameters.copy()
        if self.num_layers <= 1:
            del params['Whi']
            del params['Who']
            del params['Whc']
            del params['Whf']
        if not self.use_forget_gate:
            if 'Whf' in params:
                del params['Whf']
            del params['Wf']
            del params['Uf']
            del params['bf']
        return params.values()

    def state(self):
        state_params = {}
        for param, value in self.parameters.items():
            state_params[param] = value.get_value()
        return {
            'name': self.name,
            'n_input': self.n_input,
            'n_hidden': self.n_hidden,
            'num_layers': self.num_layers,
            'use_forget_gate': self.use_forget_gate,
            'parameters': state_params
        }

    @staticmethod
    def load(state):
        assert len(state) ==  6, state
        lstm = LSTM(state['name'], state['n_input'], state['n_hidden'],
                    num_layers=state['num_layers'],
                    use_forget_gate=state['use_forget_gate'])
        for param, value in state['parameters'].items():
            lstm.set_parameter_value(param, value)
        return lstm

if __name__ == "__main__":
    layers = 2
    O = 30
    B = 15
    D = 10
    lstm = LSTM('encoder', D, O, num_layers=layers).compile()
    X = np.ones((B, D))
    H = np.zeros((B, layers, O))
    S = np.zeros((B, layers, O))
    print lstm.step(X, H, S)
