import os
import shutil

import matplotlib
import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm

matplotlib.use('Agg')


def train(model, loader, epoch, optimizer, criterion, device, dtype, batch_size, log_interval, clip_grad=0.):
    model.train()
    correct1, correct5 = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        optimizer.zero_grad()
        loss, output = model.forward_backward(data, target, criterion)
        if clip_grad > 1e-12:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        corr = correct(model, data, output, target, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]

        if batch_idx % log_interval == 0:
            tqdm.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. '
                'Top-1 accuracy: {:.2f}%({:.2f}%). '
                'Top-5 accuracy: {:.2f}%({:.2f}%).'.format(epoch, batch_idx, len(loader),
                                                           100. * batch_idx / len(loader), loss.item(),
                                                           100. * corr[0] / batch_size,
                                                           100. * correct1 / (batch_size * (batch_idx + 1)),
                                                           100. * corr[1] / batch_size,
                                                           100. * correct5 / (batch_size * (batch_idx + 1))))
    try:
        size = len(loader.dataset)
    except:
        size = len(loader) * loader[0][0].shape[0]  # TODO
    return loss.item(), correct1 / size, correct5 / size


def adv_train(model, loader, epoch, optimizer, criterion, device, dtype, batch_size, log_interval, att, eps,
              adv_w, normalize, clip_grad=0., alpha=False, astart=1, aend=1):
    model.train()
    correct1, correct5 = 0, 0
    if alpha:
        alpha_sched = np.linspace(astart, aend, len(loader))

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        att.model = model  # TODO
        if alpha:
            model.set_alpha(alpha_sched[batch_idx])
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        optimizer.zero_grad()
        loss, loss_a, output, output_a = model.adv_forward_backward(data, target, criterion, att, eps, normalize, adv_w)
        if clip_grad > 1e-12:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        corr = correct(model, data, output, target, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]

        if batch_idx % log_interval == 0:
            tqdm.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. '
                'Top-1 accuracy: {:.2f}%({:.2f}%). '
                'Top-5 accuracy: {:.2f}%({:.2f}%).'.format(epoch, batch_idx, len(loader),
                                                           100. * batch_idx / len(loader), loss.item(),
                                                           100. * corr[0] / batch_size,
                                                           100. * correct1 / (batch_size * (batch_idx + 1)),
                                                           100. * corr[1] / batch_size,
                                                           100. * correct5 / (batch_size * (batch_idx + 1))))

            if alpha:
                tqdm.write("alpha={}".format(alpha_sched[batch_idx]))
    return loss.item(), correct1 / len(loader.dataset), correct5 / len(loader.dataset)


def attack(model, loader, criterion, att, eps, normalize, device, dtype):
    model.eval()
    test_loss = 0
    correct1, correct5 = 0, 0
    test_loss_a = 0
    correct1_a, correct5_a = 0, 0

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        x_a, output, output_a = att.generate_sample(data, target, eps, normalize)

        test_loss += criterion(output, target).item()  # sum up batch loss
        # print('o',torch.max(torch.abs(output-output_a)).item())
        corr = correct(model, data, output, target, topk=(1, 5))
        correct1 += corr[0]
        correct5 += corr[1]

        test_loss_a += criterion(output_a, target).item()  # sum up batch loss
        corr_a = correct(model, x_a, output_a, target, topk=(1, 5))
        correct1_a += corr_a[0]
        correct5_a += corr_a[1]
        # print(corr,corr_a, (x_a-data).abs().max())

    test_loss /= len(loader)
    test_loss_a /= len(loader)
    tqdm.write(
        '\nTest set: Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '
        'Top5: {}/{} ({:.2f}%)'.format(test_loss, int(correct1), len(loader.dataset),
                                       100. * correct1 / len(loader.dataset), int(correct5),
                                       len(loader.dataset), 100. * correct5 / len(loader.dataset)))
    tqdm.write(
        'Adverserial set (eps={}): Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '
        'Top5: {}/{} ({:.2f}%)'.format(eps, test_loss_a, int(correct1_a), len(loader.dataset),
                                       100. * correct1_a / len(loader.dataset), int(correct5_a),
                                       len(loader.dataset), 100. * correct5_a / len(loader.dataset)))
    return test_loss, correct1 / len(loader.dataset), correct5 / len(loader.dataset), \
           test_loss_a, correct1_a / len(loader.dataset), correct5_a / len(loader.dataset)


def gen_attack(model, loader, criterion, adv_method, eps, normalize, device, dtype):
    att = adv_method(model, criterion)
    model.eval()
    attack_set = []

    for batch_idx, (data, target) in enumerate(tqdm(loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        x_a, output, output_a = att.generate_sample(data, target, eps, normalize)
        attack_set.append((x_a, target))
    return attack_set


def test(model, loader, criterion, device, dtype):
    model.eval()
    test_loss = 0
    correct1, correct5 = 0, 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(loader)):
            data, target = data.to(device=device, dtype=dtype), target.to(device=device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            corr = correct(model, data, output, target, topk=(1, 5))
            correct1 += corr[0]
            correct5 += corr[1]

    test_loss /= len(loader)
    tqdm.write(
        '\nTest set: Average loss: {:.4f}, Top1: {}/{} ({:.2f}%), '
        'Top5: {}/{} ({:.2f}%)'.format(test_loss, int(correct1), len(loader.dataset),
                                       100. * correct1 / len(loader.dataset), int(correct5),
                                       len(loader.dataset), 100. * correct5 / len(loader.dataset)))
    return test_loss, correct1 / len(loader.dataset), correct5 / len(loader.dataset)


def correct(model, data, output, target, topk=(1,)):
    """Computes the correct@k for the specified values of k"""
    maxk = max(topk)
    pred = model.predict(data, output, maxk)
    pred = pred.t().type_as(target)
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).item()
        res.append(correct_k)
    return res


def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar'):
    save_path = os.path.join(filepath, filename)
    best_path = os.path.join(filepath, 'model_best.pth.tar')
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, best_path)
