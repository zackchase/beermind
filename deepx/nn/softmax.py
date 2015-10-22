import theano.tensor as T

from theanify import theanify

from model import ParameterModel

def Softmax(*args):
    class Softmax(ParameterModel):

        def __init__(self, name, n_input, n_output):
            super(Softmax, self).__init__(name)
            self.n_input = n_input
            self.n_output = n_output

            assert self.n_output > 1, "Need more than 1 output for softmax"

            self.init_parameter('W', self.initialize_weights((self.n_input, self.n_output)))
            self.init_parameter('b', self.initialize_weights((self.n_output,)))

        @theanify(T.matrix('X'))
        def forward(self, X):
            return T.nnet.softmax(T.dot(X, self.get_parameter('W')) + self.get_parameter('b'))

        def state(self):
            state_params = {}
            for param, value in self.parameters.items():
                state_params[param] = value.get_value()
            return {
                'n_input': self.n_input,
                'n_output': self.n_output,
                'parameters': state_params
            }


        @staticmethod
        def load(state):
            assert len(state) ==  3, state
            softmax = Softmax(state['name'], state['n_input'], state['n_output'])
            for param, value in state['parameters'].items():
                softmax.set_parameter_value(param, value)
            return softmax
    return Softmax(*args)
