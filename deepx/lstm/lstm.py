import numpy as np
import theano
import theano.tensor as T

from theanify import theanify, Theanifiable

class LSTM(Theanifiable):
    """
    W_ix    The weight matrix between input and igate
    U_ih    Tensor containing recurrent weights between each layer and corresponding layer in next time step
    b_i     Bias term for the input gate
    W_il    Weight from previous layer l to igate

    etc. for g, fgate and ogate
    """

    def __init__(self, name, n_input, n_hidden,
                 num_layers=2,
                 use_forget_gate=True,
                 use_input_peep=False, use_output_peep=False, use_forget_peep=False,
                 use_tanh_output=True, seed=None):
        super(LSTM, self).__init__()
        self.name = name
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.num_layers = num_layers
        self.use_forget_gate = use_forget_gate
        self.use_input_peep = use_input_peep
        self.use_output_peep = use_output_peep
        self.use_forget_peep = use_forget_peep
        self.use_tanh_output = use_tanh_output

        np.random.seed(seed)

        assert self.num_layers >= 1

        self.parameters = {}

        self.init_parameter('W_ix', self.initialize_weights((self.n_input, self.n_hidden)))
        self.init_parameter('U_ih', self.initialize_weights((self.num_layers, self.n_hidden, self.n_hidden)))
        self.init_parameter('b_i', self.initialize_weights((self.num_layers, self.n_hidden)))

        self.init_parameter('W_ox', self.initialize_weights((self.n_input, self.n_hidden)))
        self.init_parameter('U_oh', self.initialize_weights((self.num_layers, self.n_hidden, self.n_hidden)))
        self.init_parameter('b_o', self.initialize_weights((self.num_layers, self.n_hidden)))

        self.init_parameter('W_fx', self.initialize_weights((self.n_input, self.n_hidden)))
        self.init_parameter('U_fh', self.initialize_weights((self.num_layers, self.n_hidden, self.n_hidden)))
        self.init_parameter('b_f', self.initialize_weights((self.num_layers, self.n_hidden)))

        self.init_parameter('W_gx', self.initialize_weights((self.n_input, self.n_hidden)))
        self.init_parameter('U_gh', self.initialize_weights((self.num_layers, self.n_hidden, self.n_hidden)))
        self.init_parameter('b_g', self.initialize_weights((self.num_layers, self.n_hidden)))

        if self.use_input_peep:
            self.init_parameter('P_i', self.initialize_weights((self.num_layers, self.n_hidden, self.n_hidden)))
        if self.use_output_peep:
            self.init_parameter('P_o', self.initialize_weights((self.num_layers, self.n_hidden, self.n_hidden)))
        if self.use_forget_peep:
            self.init_parameter('P_f', self.initialize_weights((self.num_layers, self.n_hidden, self.n_hidden)))

        if self.num_layers > 1:
            self.init_parameter('W_il', self.initialize_weights((self.num_layers - 1, self.n_hidden, self.n_hidden)))
            self.init_parameter('W_ol', self.initialize_weights((self.num_layers - 1, self.n_hidden, self.n_hidden)))
            self.init_parameter('W_fl', self.initialize_weights((self.num_layers - 1, self.n_hidden, self.n_hidden)))
            self.init_parameter('W_gl', self.initialize_weights((self.num_layers - 1, self.n_hidden, self.n_hidden)))

    def initialize_weights(self, shape):
        return (np.random.rand(*shape) - 0.5) / 1000.0

    def init_parameter(self, name, value):
        assert name not in self.parameters, "Cannot re-initialize theano shared variable, use set_parameter_value"
        self.parameters[name] = theano.shared(value, name='%s-%s' % (self.name, name))

    def get_parameter(self, name):
        return self.parameters[name]

    def set_parameter_value(self, name, value):
        return self.parameters[name].set_value(value)

    def get_parameter_value(self, name):
        return self.parameters[name].get_value()

    def layer_step(self, X, previous_hidden, previous_state, layer):
        if layer == 0:
            Wi = self.get_parameter('W_ix')
            Wo = self.get_parameter('W_ox')
            if self.use_forget_gate:
                Wf = self.get_parameter('W_fx')
            Wg = self.get_parameter('W_gx')
        else:
            Wi = self.get_parameter('W_il')[layer - 1]
            Wo = self.get_parameter('W_ol')[layer - 1]
            if self.use_forget_gate:
                Wf = self.get_parameter('W_fl')[layer - 1]
            Wg = self.get_parameter('W_gl')[layer - 1]

        Ui = self.get_parameter('U_ih')[layer]
        Uo = self.get_parameter('U_oh')[layer]
        Uf = self.get_parameter('U_fh')[layer]
        Ug = self.get_parameter('U_gh')[layer]

        bi = self.get_parameter('b_i')[layer]
        bo = self.get_parameter('b_o')[layer]
        bf = self.get_parameter('b_f')[layer]
        bg = self.get_parameter('b_g')[layer]

        if self.use_input_peep:
            Pi = self.get_parameter('P_i')
            input_gate = T.nnet.sigmoid(T.dot(X, Wi) + T.dot(previous_hidden, Ui) + T.dot(previous_state, Pi) + bi)
        else:
            input_gate = T.nnet.sigmoid(T.dot(X, Wi) + T.dot(previous_hidden, Ui) + bi)
        candidate_state = T.tanh(T.dot(X, Wg) + T.dot(previous_hidden, Ug) + bg)

        if self.use_forget_gate:
            if self.use_forget_peep:
                Pf = self.get_parameter('P_f')
                forget_gate = T.nnet.sigmoid(T.dot(X, Wf) + T.dot(previous_hidden, Uf) + T.dot(previous_state, Pf) + bf)
            else:
                forget_gate = T.nnet.sigmoid(T.dot(X, Wf) + T.dot(previous_hidden, Uf) + bf)
            state = candidate_state * input_gate + previous_state * forget_gate
        else:
            state = candidate_state * input_gate + previous_state * 0

        if self.use_output_peep:
            Po = self.get_parameter('P_o')
            output_gate = T.nnet.sigmoid(T.dot(X, Wo) + T.dot(previous_hidden, Uo) + T.dot(previous_state, Po) + bo)
        else:
            output_gate = T.nnet.sigmoid(T.dot(X, Wo) + T.dot(previous_hidden, Uo) + bo)
        if self.use_tanh_output:
            output = output_gate * T.tanh(state)
        else:
            output = output_gate * state

        return output, state

    @theanify(T.matrix('X'), T.tensor3('previous_hidden'), T.tensor3('previous_state'))
    def step(self, X, previous_hidden, previous_state):
        out, state = self.layer_step(X, previous_hidden[:, 0, :], previous_state[:, 0, :], 0)
        outs = [out]
        states = [state]
        for l in xrange(1, self.num_layers):
            out, state = self.layer_step(out, previous_hidden[:, l, :], previous_state[:, l, :], l)
            states.append(state)
            outs.append(out)
        return T.swapaxes(T.stack(*outs), 0, 1), T.swapaxes(T.stack(*states), 0, 1)

    def get_theano_parameters(self):
        return self.parameters.values()

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
            'use_input_peep' : self.use_input_peep,
            'use_output_peep' : self.use_output_peep,
            'use_forget_peep' : self.use_forget_peep,
            'use_tanh_output' : self.use_tanh_output,
            'parameters': state_params
        }

    @staticmethod
    def load(state):
        assert len(state) ==  10, state
        lstm = LSTM(state['name'], state['n_input'], state['n_hidden'],
                    num_layers=state['num_layers'],
                    use_forget_gate=state['use_forget_gate'],
                    use_input_peep=state['use_input_peep'],
                    use_output_peep=state['use_output_peep'],
                    use_forget_peep=state['use_forget_peep'],
                    use_tanh_output=state['use_tanh_output'],
                    )
        for param, value in state['parameters'].items():
            lstm.set_parameter_value(param, value)
        return lstm

if __name__ == "__main__":
    layers = 2
    O = 30
    B = 15
    D = 10
    lstm = LSTM('encoder', D, O, num_layers=layers, seed=1).compile()
    X = np.ones((B, D))
    H = np.zeros((B, layers, O))
    S = np.zeros((B, layers, O))
    print lstm.step(X, H, S)
