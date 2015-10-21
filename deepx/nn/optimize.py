import theano
import theano.tensor as T

from theanify import theanify, Theanifiable

def create_optimizer(pm, *args):
    class Optimizer(Theanifiable):

        def __init__(self, parameter_model):
            super(Optimizer, self).__init__()
            self.parameter_model = parameter_model

            self.average_gradient = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]
            self.average_rms = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]
            self.parameter_update = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]

        @theanify(*args, updates="rmsprop_updates")
        def rmsprop(self, *args):
            return self.parameter_model.cost(*args)

        @theanify(*args, updates="sgd_updates")
        def sgd(self, *args):
            return self.parameter_model.cost(*args)

        def gradient(self, *args):
            return T.grad(self.parameter_model.cost(*args)[0], self.get_parameters())

        def sgd_updates(self, *args):
            grads = self.gradient(*args)
            updates = [(p, p - 0.1 * g) for p, g in zip(self.get_parameters(), grads)]
            return updates

        def rmsprop_updates(self, *args):
            grads = self.gradient(*args)
            next_average_gradient = [0.95 * avg + 0.05 * g for g, avg in zip(grads, self.average_gradient)]
            next_rms = [0.95 * rms + 0.05 * (g ** 2) for g, rms in zip(grads, self.average_rms)]
            next_parameter = [0.9 * param_update - 1e-4 * g / T.sqrt(rms - avg ** 2 + 1e-4)
                            for g, avg, rms, param_update in zip(grads,
                                                                self.average_gradient,
                                                                self.average_rms,
                                                                self.parameter_update)]

            average_gradient_update = [(avg, next_avg) for avg, next_avg in zip(self.average_gradient,
                                                                                next_average_gradient)]
            rms_update = [(rms, rms2) for rms, rms2 in zip(self.average_rms,
                                                                next_rms)]
            next_parameter_update = [(param, param_update) for param, param_update in zip(self.parameter_update,
                                                                                        next_parameter)]

            updates = [(p, p + param_update) for p, param_update in zip(self.get_parameters(), next_parameter)]

            return updates + average_gradient_update + rms_update + next_parameter_update

        def get_parameters(self):
            return self.parameter_model.get_parameters()

    return Optimizer(pm)
