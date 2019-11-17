import argparse
import csv
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import trange, tqdm

from attacks.expectation_pgd_fgsmk import EPGD_FGSMk
from attacks.fgsm import FGSM
from attacks.pgd_fgsmk import PGD_FGSMk
from logger import CsvLogger
from run import test, train, save_checkpoint, adv_train


def get_args():
    parser = argparse.ArgumentParser(description='PNI training with PyTorch')
    parser.add_argument('--data', default='./data', metavar='PATH', help='Path to data')
    parser.add_argument('--adv-data', type=str, default=None, metavar='PATH', help='Path to data')
    parser.add_argument('--dataset', default='cifar10', metavar='SET', help='Dataset (CIFAR-10, CIFAR-100)')

    parser.add_argument('--gpus', default='0', help='List of GPUs used for training - e.g 0,1,3')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')
    parser.add_argument('--print-model', action='store_true', default=False, help='print model to stdout')

    # Optimization options
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs to train.')
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--val-batch-size', default=256, type=int, metavar='N',
                        help='validation mini-batch size (default: 256)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The learning rate.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=1e-4, help='Weight decay (L2 penalty).')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
    parser.add_argument('--schedule', type=int, nargs='+', default=[100, 200, 300],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--opt', type=str, default='sgd', help='Optimizer')

    # Checkpoints
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
    parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
    parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='Number of batches between log messages')
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: random)')

    # Architecture
    parser.add_argument('--layers', type=int, default=20, metavar='L', help='Number of layers')
    parser.add_argument('--width', type=float, default=1, metavar='W', help='Width multiplier')

    # Attack
    parser.add_argument('--attack', default='fgsm', type=str, metavar='ATT', help='attack type')
    parser.add_argument('--eps', default=8 / 255, type=float, metavar='EPS', help='epsilon of attack')

    # PNI
    parser.add_argument('--pni', dest='pni', action='store_true', help='Use PNI')
    parser.add_argument('--cpni', dest='cpni', action='store_true', help='Use colored PNI')

    # Smoothing
    parser.add_argument('--smooth', default='mcpredict', type=str, metavar='SMOOTH', help='smooth type')
    # parser.add_argument('--mcpredict', dest='mcpredict', action='store_true', help='Use smooth predict')
    # parser.add_argument('--mcepredict', dest='mcepredict', action='store_true', help='Use smooth expectation predict')
    # parser.add_argument('--mceattack', dest='mceattack', action='store_true', help='Use expectation attack model')
    # parser.add_argument('--mclogits', dest='mclogits', action='store_true', help='Use smooth logits predict')
    parser.add_argument('--noise_sd', type=float, default=0.2, metavar='SD',
                        help='noise standard variation for smooth model')
    parser.add_argument('--m_train', type=int, default=8, metavar='NS',
                        help='number of monte carlo samples for smooth model while training')
    parser.add_argument('--m_test', type=int, default=8, metavar='NS',
                        help='number of monte carlo samples for smooth model while testing')
    parser.add_argument('--optimize_hyper_params', dest='optimize_hyper_params', action='store_true',
                        help='optimize hyper params')
    parser.add_argument('--check_hyper_params', dest='check_hyper_params', action='store_true',
                        help='test all specified options of hype params')
    parser.add_argument('--iterations-list', type=int, nargs='+', default=[2 ** x for x in range(11)],
                        help='m_test options to check while optimizing hyper params.')  #
    parser.add_argument('--noise-list', type=int, nargs='+', default=[x / 100 for x in range(51)],
                        help='noise_sd options to check while optimizing hyper params.')

    parser.add_argument('--weight-noise', dest='weight_noise', action='store_true', help='Use weight noise')
    parser.add_argument('--act-noise-a', dest='act_noise_a', action='store_true', help='Use activation noise A')
    parser.add_argument('--act-noise-b', dest='act_noise_b', action='store_true', help='Use activation noise B')  # TODO
    parser.add_argument('--noise-rank', type=int, default=5, metavar='R', help='Rank of colored noise')

    parser.add_argument('--alpha', dest='alpha', action='store_true', help='Use ALPHA')
    parser.add_argument('--adv', dest='adv', action='store_true', help='Use adversarial training')
    parser.add_argument('--adv-w', dest='adv_w', type=float, default=0.5, help='Adversarial example weight')
    parser.add_argument('--zero-start', dest='zero_start', action='store_true', help='Start from epoch 0')

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp
    args.save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.gpus is not None and torch.cuda.is_available():
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.enabled = True
        cudnn.benchmark = True
        args.device = 'cuda:' + str(args.gpus[0])
        torch.cuda.set_device(args.gpus[0])
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpus = []
        args.device = 'cpu'

    if args.type == 'float64':
        args.dtype = torch.float64
    elif args.type == 'float32':
        args.dtype = torch.float32
    elif args.type == 'float16':
        args.dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8

    if args.smooth == 'mcpredict':
        args.mcpredict = True
    elif args.smooth == 'mcepredict':
        args.mcepredict = True
    elif args.smooth == 'mclogits':
        args.mclogits = True
    elif args.smooth == 'mceattack':
        args.attack = True
    else:
        raise ValueError('Wrong smooth method!')

    args.num_classes = 10
    if args.dataset == 'cifar10':
        args.dataset = torchvision.datasets.CIFAR10
        args.num_classes = 10
        from cifar_data import get_loaders
    else:
        raise ValueError('Wrong dataset!')
    args.get_loaders = get_loaders

    if args.pni:
        from models.resnet_pni import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar, resnet164_cifar
    elif args.cpni:
        if args.mcpredict:
            from models.resnet_cpni_smooth_predict import resnet20_cifar, resnet32_cifar, resnet44_cifar, \
                resnet56_cifar, resnet164_cifar
        elif args.mcepredict:
            from models.resnet_cpni_smooth_expectation_predict import resnet20_cifar, resnet32_cifar, resnet44_cifar, \
                resnet56_cifar, resnet164_cifar
        elif args.mceattack:
            from models.resnet_cpni_expectation_attack import resnet20_cifar, resnet32_cifar, resnet44_cifar, \
                resnet56_cifar, resnet164_cifar
        elif args.mclogits:
            from models.resnet_cpni_smooth_logits import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar, \
                resnet164_cifar
        else:
            from models.resnet_pni_colored import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar, \
                resnet164_cifar
    else:
        from models.resnet import resnet20_cifar, resnet32_cifar, resnet44_cifar, resnet56_cifar, resnet164_cifar

    if args.layers == 20:
        args.net = resnet20_cifar
    elif args.layers == 32:
        args.net = resnet32_cifar
    elif args.layers == 44:
        args.net = resnet44_cifar
    elif args.layers == 56:
        args.net = resnet56_cifar
    elif args.layers == 164:
        args.net = resnet164_cifar
    else:
        raise ValueError('Wrong number of layers!')

    if args.attack == 'fgsm':
        args.attack = FGSM
    elif args.attack == 'pgd':
        args.attack = PGD_FGSMk
    elif args.attack == 'epgd':
        args.attack = EPGD_FGSMk
    else:
        raise ValueError('Wrong attack!')


    args.smoothing = args.mcpredict or args.mcepredict or args.mclogits
    # if args.smoothing:
    #     args.learning_rate /= args.m_train
    return args


def main():
    args = get_args()
    device, dtype = args.device, args.dtype

    add_args = {'num_classes': args.num_classes}
    if args.cpni:
        add_args = {'weight_noise': args.weight_noise, 'act_noise_a': args.act_noise_a, 'act_noise_b': args.act_noise_b,
                    'rank': args.noise_rank}
    if args.dataset == torchvision.datasets.ImageNet:
        add_args['pretrained'] = True
    else:
        add_args['width'] = args.width
    add_args['num_classes'] = args.num_classes
    smoothing_args = {}
    if args.smoothing:
        smoothing_args = {'noise_sd': args.noise_sd, 'm_test': args.m_test, 'm_train': args.m_train}

    model = args.net(**smoothing_args, **add_args)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    print(model)
    print("Number of parameters {}".format(num_parameters))

    train_loader, val_loader, adv_data = args.get_loaders(args.dataset, args.data, args.batch_size, args.val_batch_size,
                                                          args.workers, args.adv_data)
    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    model, criterion = model.to(device=device, dtype=dtype), criterion.to(device=device, dtype=dtype)

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.decay, nesterov=True)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.decay)
    else:
        raise ValueError('Wrong optimzier!')

    scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

    best_test = 0

    # optionally resume from a checkpoint
    data = None
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch'] - 1
            best_test = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        elif os.path.isdir(args.resume):
            checkpoint_path = os.path.join(args.resume, 'checkpoint.pth.tar')
            csv_path = os.path.join(args.resume, 'results.csv')
            print("=> loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location=device)
            args.start_epoch = checkpoint['epoch'] - 1
            best_test = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
            data = []
            with open(csv_path) as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    data.append(row)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.zero_start:
        args.start_epoch = 0
    if args.evaluate:
        loss, top1, top5 = test(model, val_loader, criterion, device, dtype)  # TODO
        return

    csv_logger = CsvLogger(filepath=args.save_path, data=data)
    csv_logger.save_params(sys.argv, args)

    claimed_acc1 = None
    claimed_acc5 = None
    if args.adv:
        normalize = {'mean': np.array([0.491, 0.482, 0.447]), 'std': np.array([0.247, 0.243, 0.262])}
        if args.alpha:
            adv_train_network_alpha(args.start_epoch, args.epochs, scheduler, model, train_loader, val_loader,
                                    optimizer, criterion, device, dtype, args.batch_size, args.log_interval, csv_logger,
                                    args.save_path, claimed_acc1, claimed_acc5, best_test, args.attack, args.eps, 0.5,
                                    normalize)
        else:
            a = smoothing_args
            a.update(add_args)
            a['width'] = args.width
            adv_train_network(args.start_epoch, args.epochs, scheduler, model, train_loader, val_loader, optimizer,
                              criterion, device, dtype, args.batch_size, args.log_interval, csv_logger, args.save_path,
                              claimed_acc1, claimed_acc5, best_test, args.attack, args.eps, args.adv_w, normalize, args,
                              a)
    else:
        train_network(args.start_epoch, args.epochs, scheduler, model, train_loader, val_loader, adv_data, optimizer,
                      criterion, device, dtype, args.batch_size, args.log_interval, csv_logger, args.save_path,
                      claimed_acc1, claimed_acc5, best_test)


def train_network(start_epoch, epochs, scheduler, model, train_loader, val_loader, adv_data, optimizer, criterion,
                  device, dtype, batch_size, log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5,
                  best_test):
    for epoch in trange(start_epoch, epochs + 1):
        train_loss, train_accuracy1, train_accuracy5, = train(model, train_loader, epoch, optimizer, criterion, device,
                                                              dtype, batch_size, log_interval)
        if adv_data is not None:
            traina_loss, traina_accuracy1, traina_accuracy5, = train(model, adv_data, epoch, optimizer, criterion,
                                                                     device, dtype, batch_size, log_interval)
        test_loss, test_accuracy1, test_accuracy5 = test(model, val_loader, criterion, device, dtype)
        csv_logger.write({'epoch': epoch + 1, 'val_error1': 1 - test_accuracy1, 'val_error5': 1 - test_accuracy5,
                          'val_loss': test_loss, 'train_error1': 1 - train_accuracy1,
                          'train_error5': 1 - train_accuracy5, 'train_loss': train_loss})
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
                         'optimizer': optimizer.state_dict()}, test_accuracy1 > best_test, filepath=save_path)

        csv_logger.plot_progress(claimed_acc1=claimed_acc1, claimed_acc5=claimed_acc5)

        if test_accuracy1 > best_test:
            best_test = test_accuracy1
        for layer in model.modules():
            from layers import NoisedConv2D, NoisedLinear
            if isinstance(layer, NoisedConv2D) or isinstance(layer, NoisedLinear):
                print("Mean of alphas is {}".format(torch.mean(layer.alpha)))
        scheduler.step()

    csv_logger.write_text('Best accuracy is {:.2f}% top-1'.format(best_test * 100.))


def adv_train_network(start_epoch, epochs, scheduler, model, train_loader, val_loader, optimizer, criterion, device,
                      dtype, batch_size, log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5, best_test,
                      adv_method, eps, adv_w, normalize, args, subts_args=None):
    att_object = adv_method(model, criterion)
    for epoch in trange(start_epoch, epochs + 1):

        train_loss, train_accuracy1, train_accuracy5, = adv_train(model, train_loader, epoch, optimizer, criterion,
                                                                  device, dtype, batch_size, log_interval, att_object,
                                                                  eps, adv_w, normalize, 0.05)
        test_loss, test_accuracy1, test_accuracy5 = test(model, val_loader, criterion, device, dtype)
        csv_logger.write({'epoch': epoch + 1, 'val_error1': 1 - test_accuracy1, 'val_error5': 1 - test_accuracy5,
                          'val_loss': test_loss, 'train_error1': 1 - train_accuracy1,
                          'train_error5': 1 - train_accuracy5, 'train_loss': train_loss})
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
                         'optimizer': optimizer.state_dict()}, test_accuracy1 > best_test, filepath=save_path)

        csv_logger.plot_progress(claimed_acc1=claimed_acc1, claimed_acc5=claimed_acc5)

        if test_accuracy1 > best_test:
            best_test = test_accuracy1
        for layer in model.modules():
            from layers import NoisedConv2D, NoisedLinear, NoisedConv2DColored
            if isinstance(layer, NoisedConv2D) or isinstance(layer, NoisedLinear):
                tqdm.write("Mean of alphas is {}".format(torch.mean(layer.alpha)))
            if isinstance(layer, NoisedConv2DColored):
                try:

                    tqdm.write("Mean of alphas_diag_w is {}+-{} ({}) ".format(torch.mean(torch.abs(layer.alphad_w)),
                                                                              torch.std(torch.abs(layer.alphad_w)),
                                                                              torch.max(torch.abs(layer.alphad_w))))
                    tqdm.write("Mean of alphas_factor_w is {}+-{} ({}) ".format(torch.mean(torch.abs(layer.alphaf_w)),
                                                                                torch.std(layer.alphaf_w),
                                                                                torch.max(torch.abs(layer.alphaf_w))))
                except:
                    pass

                try:
                    tqdm.write("Mean of alphas_diag_a is {}+-{} ({})  ".format(torch.mean(torch.abs(layer.alphad_i)),
                                                                               torch.std(torch.abs(layer.alphad_i)),
                                                                               torch.max(torch.abs(layer.alphad_i))))
                    tqdm.write("Mean of alphas_factor_a is {}+-{} ({}) ".format(torch.mean(torch.abs(layer.alphaf_i)),
                                                                                torch.std(layer.alphaf_i),
                                                                                torch.max(torch.abs(layer.alphaf_i))))
                except:
                    pass
        scheduler.step()

    csv_logger.write_text('Best accuracy is {:.2f}% top-1'.format(best_test * 100.))


def adv_train_network_alpha(start_epoch, epochs, scheduler, model, train_loader, val_loader, optimizer, criterion,
                            device,
                            dtype, batch_size, log_interval, csv_logger, save_path, claimed_acc1, claimed_acc5,
                            best_test,
                            adv_method, eps, adv_w, normalize):
    # alpha_sched = np.concatenate((np.ones(epochs // 8), np.linspace(1, 0, epochs - 2 * (epochs // 8)), np.zeros(epochs // 8)))
    alpha_sched = np.concatenate(
        (np.ones(epochs // 8), np.logspace(0, -4, epochs - 2 * (epochs // 8)), np.zeros(epochs // 8 + 20)))
    for epoch in trange(start_epoch, epochs + 1):
        model.set_alpha(alpha_sched[epoch])
        tqdm.write("alpha={}".format(alpha_sched[epoch]))
        train_loss, train_accuracy1, train_accuracy5, = adv_train(model, train_loader, epoch, optimizer, criterion,
                                                                  device, dtype, batch_size, log_interval, adv_method,
                                                                  eps, adv_w, normalize, 0.05, True, alpha_sched[epoch],
                                                                  alpha_sched[epoch + 1])
        test_loss, test_accuracy1, test_accuracy5 = test(model, val_loader, criterion, device, dtype)
        csv_logger.write({'epoch': epoch + 1, 'val_error1': 1 - test_accuracy1, 'val_error5': 1 - test_accuracy5,
                          'val_loss': test_loss, 'train_error1': 1 - train_accuracy1,
                          'train_error5': 1 - train_accuracy5, 'train_loss': train_loss})
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
                         'optimizer': optimizer.state_dict()}, test_accuracy1 > best_test, filepath=save_path)

        csv_logger.plot_progress(claimed_acc1=claimed_acc1, claimed_acc5=claimed_acc5)

        if test_accuracy1 > best_test:
            best_test = test_accuracy1
        scheduler.step()

    csv_logger.write_text('Best accuracy is {:.2f}% top-1'.format(best_test * 100.))


if __name__ == '__main__':
    main()
