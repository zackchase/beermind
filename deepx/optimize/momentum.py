from optimizer import Optimizer

class SGD(Optimizer):

    def __init__(self, parameter_model, args, training_rate=0.001):
        self.training_rate = training_rate
        self.parameter_model = parameter_model
        self.rho = .1
        self.caches = [theano.shared(p.get_value() * 0) for p in self.get_parameters()]

        super(SGD, self).__init__(parameter_model, args)

    def updates(self, *args):
        updates = []
        for p, c, g in zip(self.get_parameters(), self.caches, self.grads):
            delta = slef.rho * g + (1-self.rho) * c
            updates.append([c, delta])
            updates.append([p, p - eta * delta])
        return updates
