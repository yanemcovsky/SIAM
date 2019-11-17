import torch

from attacks.attack import Attack


class FGSM(Attack):
    def __init__(self, net, loss):
        super(FGSM, self).__init__(net, loss)

    def generate_sample(self, x, y, eps, normalize):
        self.net.eval()
        x.requires_grad = True
        self.net.zero_grad()
        loss, output = self.net.forward_backward(x, y, self.loss)

        epses = eps / normalize['std']
        e_ten = torch.from_numpy(epses).view(1, 3, 1, 1).to(x.device, x.dtype)
        mins = (0 - normalize['mean']) / normalize['std']
        maxs = (1 - normalize['mean']) / normalize['std']

        x_a = x.clone().detach() + x.grad.sign() * e_ten
        for i in range(3):
            x_a[:, i, :, :] = torch.clamp(x_a[:, i, :, :], mins[i], maxs[i])

        with torch.no_grad():
            o = self.net(x)
            oa = self.net(x_a)
        return x_a.detach(), o, oa
