import theano
from optimizer import Optimizer

class Momentum(Optimizer):

    def __init__(self, parameter_model, args, eta=0.001, rho=0.1):
        self.args = args
        self.parameter_model = parameter_model

        self.eta = eta
        self.rho = rho
        self.caches = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]

        super(Momentum, self).__init__(parameter_model, args)

    def updates(self, *args):
        updates = []
        for p, c, g in zip(self.get_parameters(), self.caches, self.grads):
            delta = self.rho * g + (1-self.rho) * c
            updates.append((c, delta))
            updates.append((p, p - self.eta * delta))
        return updates
