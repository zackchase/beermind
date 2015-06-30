import theano
import numpy as np

from theanify import Theanifiable

class ParameterModel(Theanifiable):

    def __init__(self, name):
        super(ParameterModel, self).__init__()
        self.name = name
        self.parameters = {}

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

    def get_parameters(self):
        return self.parameters.values()

