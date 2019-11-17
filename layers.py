import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://math.stackexchange.com/questions/2848517/sampling-multivariate-normal-with-low-rank-covariance
def lowrank_multivariate_sample(mean, factor, diag):
    event_shape = factor.shape[1]
    rank_shape = factor.shape[2]
    noise_d = torch.normal(mean=torch.zeros(event_shape).to(factor), std=1)
    if rank_shape > 0:
        noise_f = torch.normal(mean=torch.zeros(rank_shape).to(factor), std=1)
        return mean + torch.sqrt(torch.abs(diag)) * noise_d + torch.matmul(factor, noise_f)
    else:
        return mean + torch.sqrt(torch.abs(diag)) * noise_d


class NoisedConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 weight_noise=True, act_noise_a=False, act_noise_b=False, act_dim_a=None, act_dim_b=None):
        super(NoisedConv2D, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                           bias)
        self.weight_noise = weight_noise
        self.act_noise_a = act_noise_a
        self.act_noise_b = act_noise_b
        if self.weight_noise:
            self.alpha = nn.Parameter(torch.ones_like(self.weight) * .25)
        if self.act_noise_a:
            if act_dim_a is None:
                raise ValueError('Dimension of activation should be specified')
            shape_i = (self.weight.shape[1], act_dim_a, act_dim_a)
            self.alpha_i = nn.Parameter(torch.ones(shape_i) * .25)
        if self.act_noise_b:
            if act_dim_b is None:
                raise ValueError('Dimension of activation should be specified')
            shape_o = (self.weight.shape[0], act_dim_a, act_dim_a)
            self.alpha_o = nn.Parameter(torch.ones(shape_o) * .25)

    def forward(self, input):
        if self.act_noise_a:
            noise_i = torch.normal(mean=torch.zeros_like(input), std=torch.std(input))
            input = input + self.alpha_i * noise_i
        if self.weight_noise:
            noise_w = torch.normal(mean=torch.zeros_like(self.weight), std=torch.std(self.weight))
            out = F.conv2d(input, self.weight + self.alpha * noise_w, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)
        else:
            out = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.act_noise_b:
            noise_o = torch.normal(mean=torch.zeros_like(out), std=torch.std(out))
            out = out + self.alpha_o * noise_o
        return out


class NoisedConv2DColored(nn.Conv2d):  # TODO: parametrize
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 weight_noise=True, act_noise_a=False, act_noise_b=False, act_dim_a=None, act_dim_b=None, rank=5):
        super(NoisedConv2DColored, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                                                  groups, bias)
        self.weight_noise = weight_noise
        self.act_noise_a = act_noise_a
        self.act_noise_b = act_noise_b
        self.act_dim_a = act_dim_a
        self.act_dim_b = act_dim_b

        self.rank = rank
        self.noised_strength = 0.25 ** 2  # variance
        self.noisef_strength = 0.01  # covariance
        self.anoised_strength = self.noised_strength
        self.wnoised_strength = self.noised_strength
        self.anoisef_strength = self.noisef_strength
        self.wnoisef_strength = self.noisef_strength

        self.w_size = np.prod(self.weight.shape)
        if self.weight_noise:
            self.alphad_w = nn.Parameter(torch.ones(self.w_size) * self.wnoised_strength)
            self.alphaf_w = nn.Parameter(torch.ones((1, self.w_size, self.rank)) * self.wnoisef_strength)
        if self.act_noise_a:
            if act_dim_a is None:
                raise ValueError('Dimension of activation should be specified')
            self.i_size = self.weight.shape[1] * act_dim_a * act_dim_a
            self.alphad_i = nn.Parameter(torch.ones(self.i_size) * self.anoised_strength)
            self.alphaf_i = nn.Parameter(torch.ones((1, self.i_size, self.rank)) * self.anoisef_strength)
        if self.act_noise_b:
            if act_dim_b is None:
                raise ValueError('Dimension of activation should be specified')
            self.o_size = self.weight.shape[0] * act_dim_b * act_dim_b
            self.alphad_o = nn.Parameter(torch.ones(self.o_size) * self.anoised_strength)
            self.alphaf_o = nn.Parameter(torch.ones((1, self.o_size, self.rank)) * self.anoisef_strength)

    def forward(self, input):
        if self.act_noise_a:
            m = lowrank_multivariate_sample(torch.zeros(self.i_size).to(input),
                                            self.alphaf_i,
                                            torch.std(input) * self.alphad_i)
            input = input + m.view_as(input[0, :])
        if self.weight_noise:
            m = lowrank_multivariate_sample(torch.zeros(self.w_size).to(self.weight),
                                            self.alphaf_w,
                                            torch.std(self.weight) * self.alphad_w)
            out = F.conv2d(input, self.weight + m.view_as(self.weight), self.bias,
                           self.stride, self.padding, self.dilation, self.groups)
        else:
            out = F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.act_noise_b:
            m = lowrank_multivariate_sample(torch.zeros(self.o_size).to(out),
                                            self.alphaf_o,
                                            torch.std(out) * self.alphad_o)
            out = out + m.view_as(out[0, :])
        assert torch.all(torch.isfinite(out))
        return out


class NoisedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(NoisedLinear, self).__init__(in_features, out_features, bias)
        self.alpha = nn.Parameter(torch.ones_like(self.weight) * .25)

    def forward(self, input):
        noise = torch.normal(mean=torch.zeros_like(self.weight), std=torch.std(self.weight))
        return F.linear(input, self.weight + self.alpha * noise, self.bias)
