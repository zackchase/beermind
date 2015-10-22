from optimizer import Optimizer

class SGD(Optimizer):

    def __init__(self, parameter_model, args, training_rate=0.001):
        self.training_rate = training_rate

        super(SGD, self).__init__(parameter_model, args)

    def updates(self, *args):
        return [(p, p - self.training_rate * g) for p, g in zip(self.get_parameters(), self.grads)]
