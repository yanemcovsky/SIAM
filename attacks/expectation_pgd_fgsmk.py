import torch

from attacks.attack import Attack


class EPGD_FGSMk(Attack):
    def __init__(self, net, loss, k=7, rand=True):
        super(EPGD_FGSMk, self).__init__(net, loss)
        self.k = k
        self.rand = rand

    def generate_sample(self, x, y, eps, normalize):
        epses = eps / normalize['std']
        e_ten = torch.from_numpy(epses).view(1, 3, 1, 1).to(x.device, x.dtype)
        a_ten = e_ten / 4  # TODO
        mins = (0 - normalize['mean']) / normalize['std']
        maxs = (1 - normalize['mean']) / normalize['std']

        self.net.eval()
        x_i = x.clone().detach()
        if self.rand:
            for i in range(3):
                x_i[:, i, :, :] = x_i[:, i, :, :] + torch.empty_like(x_i[:, i, :, :]).uniform_(-epses[i], epses[i])
        x_i.requires_grad = True

        for j in range(self.k):
            self.net.zero_grad()

            loss, output = self.net.expectation_forward_backward(x_i, y, self.loss)

            x_i.data += x_i.grad.sign() * a_ten
            for i in range(3):
                x_i.data[:, i, :, :] = torch.max(torch.min(x_i[:, i, :, :], x[:, i, :, :] + epses[i]),
                                                 x[:, i, :, :] - epses[i])
            for i in range(3):
                x_i.data[:, i, :, :] = torch.clamp(x_i[:, i, :, :], mins[i], maxs[i])
        with torch.no_grad():
            o = self.net.forward(x)
            oi = self.net.forward(x_i)
        return x_i.detach(), o, oi
