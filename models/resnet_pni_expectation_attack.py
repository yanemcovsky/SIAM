'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

import math

import torch
import torch.nn as nn

from layers import NoisedConv2D as Conv2d
from layers import NoisedLinear as Linear
from tqdm import trange
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    " 3x3 convolution with padding "
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, width=1, num_classes=10, noise_sd=0.0, m_test=1, m_train=1, learn_noise=False):
        super(ResNet_Cifar, self).__init__()
        inplanes = int(16 * width)
        self.inplanes = inplanes
        self.conv1 = Conv2d(3, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, 2 * inplanes, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * inplanes, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = Linear(4 * inplanes * block.expansion, num_classes)
        self.num_classes = num_classes
        self.learn_noise = learn_noise
        self.noise_sd = torch.tensor(noise_sd, requires_grad=learn_noise)
        self.m_test = m_test
        self.m_train = m_train

        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward_backward(self, data, target, criterion, factor=1.):
        output = self(data)

        loss = criterion(output, target)
        loss.backward()
        return loss, output

    def expectation_forward_backward(self, data, target, criterion, factor=1.):
        if self.m_train < 2:
            # x_sample = data + torch.randn_like(data) * self.noise_sd
            # output = self.forward(x_sample, add_noise=False)
            output = self.forward(data)
            loss = criterion(output, target)
            loss.backward()
            return loss, output
        output_list = []
        loss_list = []
        factor = factor / self.m_train
        for _ in trange(self.m_train):
            # x_sample = data + torch.randn_like(data) * self.noise_sd
            # output_i = self.forward(x_sample, add_noise=False)
            output_i = self.forward(data)
            loss_i = factor * criterion(output_i, target)
            loss_i.backward()

            output_list.append(output_i.detach().unsqueeze(0))
            loss_list.append(loss_i.detach().unsqueeze(0))

        outs = torch.cat(output_list)
        output = outs.mean(0)
        losses = torch.cat(loss_list)
        loss = losses.sum(0)
        return loss, output

    def adv_forward_backward(self, data, target, criterion, att, eps, normalize, adv_w):
        self.eval()
        data_a, _, _ = att.generate_sample(data, target, eps, normalize=normalize)

        self.zero_grad()
        self.train()
        output = self(data)
        output_a = self(data_a)  # TODO: optimize

        ad_loss = criterion(output_a, target)
        reg_loss = criterion(output, target)
        loss = adv_w * ad_loss + (1 - adv_w) * reg_loss
        # tqdm.write("Reg {} Ad{} Tot {}".format(ad_loss.item(), reg_loss.item(), loss.item()))
        loss.backward()
        return loss, ad_loss, output, output_a

    def predict(self, x, output, maxk):
        _, pred = output.topk(maxk, 1, True, True)
        # self.zero_grad()
        # return pred
        return pred

    # def monte_carlo_predict(self, x, maxk, pred):
    #     # print("x.shape")
    #     # print(x.shape)
    #     # print("maxk")
    #     # print(maxk)
    #     # print("pred.shape")
    #     # print(pred.shape)
    #     # x_sample = x + torch.randn_like(x) * self.noise_sd
    #     # output = self.forward(x_sample)
    #     # _, new_pred = output.topk(maxk, 1, True, True)
    #     # return new_pred
    #     # if self.noise_sd == 0.0:
    #     #     return pred
    #     # print("self.noise_sd")
    #     # print(self.noise_sd)
    #
    #     # self.eval()
    #     with torch.no_grad():
    #         if self.m_test == 1:
    #             # x_sample = x + torch.randn_like(x) * torch.abs(self.noise_sd)
    #             # x_sample = x + torch.randn_like(x) * self.noise_sd
    #             # output = self.forward(x_sample, add_noise=False)
    #             output = self.forward(x)
    #             # output = self.forward(x_sample)
    #             _, predictions = output.topk(maxk, 1, True, True)
    #             return predictions
    #         pred_flat = pred.view(-1)
    #
    #         histogram = torch.zeros(pred_flat.shape[0], self.num_classes).to(x)
    #
    #         for _ in trange(self.m_test):
    #             # print("for i in trange(self.m_test):")
    #             # x_sample = x + torch.randn_like(x) * torch.abs(self.noise_sd)
    #             # x_sample = x + torch.randn_like(x) * self.noise_sd
    #             # print("x_sample = x + torch.randn_like(x) * self.noise_sd")
    #             # output = self.forward(x_sample, add_noise=False)
    #             output = self.forward(x)
    #             # output = output.detach()
    #             # self.zero_grad()
    #
    #             # output = self.forward(x_sample)
    #             # print("output = self.forward(x_sample)")
    #             _, pred_i = output.topk(maxk, 1, True, True)
    #             # print("_, pred_i = output.topk(maxk, 1, True, True)")
    #             pred_i_flat = pred_i.view(-1, 1)
    #             # print("pred_i_flat = pred_i.view(-1, 1)")
    #             histogram_values = histogram.gather(1, pred_i_flat)
    #             # print(" histogram_values = histogram.gather(1, pred_i_flat)")
    #             histogram = histogram.scatter(1, pred_i_flat, histogram_values + 1)
    #             # print("histogram = histogram.scatter(1, pred_i_flat, histogram_values + 1)")
    #         histogram = histogram.view(pred.shape[0], pred.shape[1], self.num_classes)
    #         predict = torch.empty_like(pred)
    #
    #         for j in range(maxk):
    #             predict[:, j] = torch.argmax(histogram[:, j, :], dim=1)
    #             histogram[np.arange(histogram.shape[0]), :, predict[:, j]] = -1
    #
    #         return predict

    def to(self, *args, **kwargs):
        super(ResNet_Cifar, self).to(*args, **kwargs)
        self.noise_sd = self.noise_sd.to(*args, **kwargs).detach().requires_grad_(self.learn_noise)
        return self

        # model.eval()
        # if model.m_test == 1:
        #     x_sample = x + torch.randn_like(x) * model.noise_sd
        #     output = model.forward(x_sample)
        #     _, predictions = output.topk(maxk, 1, True, True)
        #     return predictions
        # pred_flat = pred.view(-1)
        # # print("pred_flat.shape")
        # # print(pred_flat.shape)
        # histogram = torch.zeros(pred_flat.shape[0], model.num_classes).cuda()
        # # print("histogram.shape")
        # # print(histogram.shape)
        #
        # # histogram = [[0 for c in range(model.num_classes)] for j in range(maxk)]
        # for i in range(model.m_test):
        #     x_sample = x + torch.randn_like(x) * model.noise_sd
        #     output = model.forward(x_sample)
        #     _, pred_i = output.topk(maxk, 1, True, True)
        #     pred_i_flat = pred_i.view(-1, 1)
        #     histogram_values = histogram.gather(1, pred_i_flat)
        #     histogram = histogram.scatter(1, pred_i_flat, histogram_values + 1)
        # histogram = histogram.view(pred.shape[0], pred.shape[1], model.num_classes)
        # predict = torch.empty_like(pred)
        #
        # for j in range(maxk):
        #     predict[:, j] = torch.argmax(histogram[:, j, :], dim=1)
        #     histogram[np.arange(histogram.shape[0]), :, predict[:, j]] = -1
        #
        # return predict


class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = Linear(64 * block.expansion, num_classes)
        self.num_classes = num_classes
        self.noise_sd = 0.0
        self.m_train = 1
        self.m_test = 1

        for m in self.modules():
            if isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def forward_backward(self, data, target, criterion, factor=1.):
        output = self(data)

        loss = criterion(output, target)
        loss.backward()
        return loss, output

    def adv_forward_backward(self, data, target, criterion, att, eps, normalize, adv_w):
        self.eval()
        data_a, _, _ = att.generate_sample(data, target, eps, normalize=normalize)

        self.zero_grad()
        self.train()
        output = self(data)
        output_a = self(data_a)  # TODO: optimize

        ad_loss = criterion(output_a, target)
        reg_loss = criterion(output, target)
        loss = adv_w * ad_loss + (1 - adv_w) * reg_loss
        # tqdm.write("Reg {} Ad{} Tot {}".format(ad_loss.item(), reg_loss.item(), loss.item()))
        loss.backward()
        return loss, ad_loss, output, output_a

    def predict(self, x, output, maxk):
        _, pred = output.topk(maxk, 1, True, True)
        return pred


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model


if __name__ == '__main__':
    net = resnet20_cifar()
    y = net(torch.randn(1, 3, 32, 32))
    print(net)
    print(y.size())
