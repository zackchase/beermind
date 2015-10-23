import theano.tensor as T

from theanify import theanify, Theanifiable

class Optimizer(Theanifiable):

    def __init__(self, parameter_model, optimize_args=[]):
        super(Optimizer, self).__init__()
        self.parameter_model = parameter_model
        self.cost_args = self.parameter_model.cost.args
        self.optimize_args = optimize_args

        self.cost, self.state = self.parameter_model.cost(*self.cost_args)
        self.grads = T.grad(self.cost, self.get_parameters())
        self.compile_method('optimize', args=self.optimize_args + list(self.cost_args))

    @theanify(updates="updates")
    def optimize(self, *args):
        return self.cost, self.state

    def get_parameters(self):
        return self.parameter_model.get_parameters()
