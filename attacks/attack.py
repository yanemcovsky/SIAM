class Attack:
    def __init__(self, net, loss):
        self.net = net
        self.loss = loss

    def generate_sample(self, x, y, eps, normalize):
        # TODO: move normalization into network
        raise NotImplementedError('You need to define a generate_sample method!')
