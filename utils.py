import datetime
import os
import wandb
import argparse
import random
import numpy as np
import torch
import plotly.express as px
import matplotlib.pyplot as plt
import time


from collections import defaultdict, deque
from torch import optim
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment as linear_assignment
from functools import partial
from sklearn.manifold import TSNE
# from medmnist import DermaMNIST

from models.wrn_mixup_model import wrn28_10
from models.res_mixup_model import resnet18
from models.resnet12 import Res12
from models.resnet_ssl import resnet12_ssl
from datasets import MiniImageNet, CIFAR, CUB, ISIC2018, BreakHis, PapSmear, Blood, DermaMNIST, OrganAMNIST, PathMNIST
from datasets.samplers import CategoriesSampler
from methods import PTMAPLoss, ProtoLoss
from self_optimal_transport import SOT

try:
    import wandb
    HAS_WANDB = True
except Exception as e:
    HAS_WANDB = False

models = dict(wrn=wrn28_10, resnet18=resnet18, resnet12=Res12, resnet_ssl=resnet12_ssl)
datasets = dict(miniimagenet=MiniImageNet, cifar=CIFAR, isic2018=ISIC2018, breakhis=BreakHis, papsmear=PapSmear, blood=Blood, dermamnist=DermaMNIST, organamnist=OrganAMNIST,pathmnist=PathMNIST
                )
n_cls = dict(isic2018=7, breakhis=8, papsmear=7, blood=11, pathmnist=9, dermamnist=7, organamnist=11)
methods = dict(pt_map=PTMAPLoss, pt_map_sot=PTMAPLoss, proto=ProtoLoss, proto_sot=ProtoLoss, )
num_base_cls = dict(isic2018=4, breakhis=5, papsmear=4)

def get_model(model_name: str, m: int,  args):
    """
    Get the backbone model.
    """
    arch = model_name.lower()
    if arch in models.keys():
        if 'vit' in arch:
            model = models[arch](num_classes=n_cls[args.dataset.lower()])
        elif(arch.endswith('ssl')):
            model = models[arch](avg_pool=True, drop_rate=0.1, dropblock_size=5, num_classes=n_cls[args.dataset.lower()])
        elif(arch.startswith('resnet18')):
            model = models[arch](num_classes=n_cls[args.dataset.lower()])
        else:
            model = models[arch](num_classes=n_cls[args.dataset.lower()], num_sla=m, dropRate=args.dropout)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

        return model
    else:
        raise ValueError(f'Model {model_name} not implemented. available models are: {list(models.keys())}')

def get_sla_classifier(args, model, m):
    modules = []
    output_types = [n_cls[args.dataset.lower()] * m, n_cls[args.dataset.lower()]]

    for out in output_types:
        if type(out) is int:
            modules.append(torch.nn.Linear(640, out))
        else:
            raise Exception('out should be integer')
        
    model = sla_pred_model(model, modules)

    return model


def get_dataloader(set_name: str, args: argparse, constant: bool = False):
    """
    Get dataloader with categorical sampler for few-shot classification.
    """
    num_episodes = args.set_episodes[set_name]
    num_way = args.train_way if set_name == 'train' else args.val_way

    """
    if 'mnist' in args.dataset.lower():
        mean = [x / 255.0 for x in [129.37731888, 124.10583864, 112.47758569]]
        std = [x / 255.0 for x in [68.20947949, 65.43124043, 70.45866994]]
        normalize = transforms.Normalize(mean=mean, std=std)
        
        if set_name == 'train':
            transforms_list = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
            ]
        else:
            transforms_list = [
                transforms.ToTensor(),
            ]

        transform = transforms.Compose(
            transforms_list + [normalize]
        )

        data_set = datasets[args.dataset.lower()](
            split=set_name, transform=transform, download=True, as_rgb=True, size=args.img_size
        )
    else:
    """
    # define dataset sampler and data loader
    data_set = datasets[args.dataset.lower()](
        args.data_path, set_name, args.backbone, augment=set_name == 'train' and args.augment
    )

    data_sampler = CategoriesSampler(
        set_name, data_set.labels, num_episodes, const_loader=constant,
        num_way=num_way, num_shot=args.num_shot, num_query=args.num_query
    )
    return DataLoader(
        data_set, batch_sampler=data_sampler, num_workers=args.num_workers, pin_memory=not constant,
    )


def get_optimizer(args: argparse, params):
    """
    Get optimizer.
    """
    if args.optimizer == 'adam':
        return optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        return optim.SGD(params, lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Optimizer {args.optimizer} not available.')


def get_scheduler(args: argparse, optimizer: torch.optim):
    """
    Get optimizer.
    """
    if args.scheduler is None:
        return None
    elif args.scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        raise ValueError(f'Scheduler {args.scheduler} not available.')


def get_method(args: argparse, sot: SOT):
    """
    Get the few-shot classification method (e.g. pt_map).
    """

    if args.method.lower() in methods.keys():
        return methods[args.method.lower()](args=vars(args), sot=sot)
    else:
        raise ValueError(f'Not implemented method. available methods are: {methods.keys()}')


def get_criterion_by_method(method: str):
    """
    Get loss function based on the method.
    """

    if 'pt_map' in method:
        return torch.nn.NLLLoss()
    elif 'proto' in method:
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Not implemented criterion for this method. available methods are: {list(methods.keys())}')


def get_criterion_by_backbone(backbone: str):
    """
    Get loss function based on the backbone.
    """

    if 'wrn' or 'resnet' in backbone:
        return torch.nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Not implemented criterion for this backbone. available methods are: {list(models.keys())}')


def load_criterion_optimizer(criterion, optimizer, args):
    pretrained_path = args.pretrained_path

    if not pretrained_path or 'miniImagenet' in pretrained_path or args.eval is True:
        return criterion, optimizer

    print(f'Loading criterion and optimizer from {pretrained_path}')

    checkpoint = torch.load(pretrained_path)

    criterion.load_state_dict(checkpoint['criterion_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("Criterion and optimizer load successfully ")
    return criterion, optimizer


def init_wandb(exp_name: str, args):
    """
    Initialize and returns wandb logger if args.wandb is True.
    """
    if not args.wandb:
        return None
    assert HAS_WANDB, "Install wandb via - 'pip install wandb' in order to use wandb logging. "
    logger = wandb.init(project=args.project, name=exp_name, config=vars(args))
    # define which metrics will be plotted against it
    logger.define_metric("train_loss", step_metric="epoch")
    logger.define_metric("train_accuracy", step_metric="epoch")
    logger.define_metric("val_loss", step_metric="epoch")
    logger.define_metric("val_accuracy", step_metric="epoch")
    return logger


def wandb_log(results: dict):
    """
    Log step to the logger without print.
    """
    if HAS_WANDB and wandb.run is not None:
        wandb.log(results)


# def get_logger(exp_name: str, args: argparse):
#     """
#     Initialize and returns wandb logger if args.wandb is True.
#     """
#     if args.wandb:
#         logger = wandb.init(project=args.project, entity=args.entity, name=exp_name, config=vars(args))
#         # define which metrics will be plotted against it
#         logger.define_metric("train_loss", step_metric="epoch")
#         logger.define_metric("train_accuracy", step_metric="epoch")
#         logger.define_metric("val_loss", step_metric="epoch")
#         logger.define_metric("val_accuracy", step_metric="epoch")
#         return logger

#     return None


# def log_step(results: dict, logger: wandb):
#     """
#     Log step to the logger without print.
#     """
#     if logger is not None:
#         logger.log(results)

#     for key, value in results.items():
#         if 'acc' in key:
#             print(f"{key}: {100 * value:.2f}%")
#         else:
#             print(f"{key}: {value:.4f}")


def get_output_dir(args: argparse):
    """
    Initialize the output dir.
    """

    out_dir = f'./checkpoints/{args.backbone.lower()}/{args.dataset.lower()}/' \
              f'way{args.train_way}_shot{args.num_shot}' \
              f'_lr{args.lr}_drop{args.dropout}'

    if args.eval:
        return out_dir

    while os.path.exists(out_dir):
        out_dir += f'_{np.random.randint(100)}'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # write args to a file
    with open(os.path.join(out_dir, "args.txt"), 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))

    print("Model checkpoints will be saved at:", out_dir)
    return out_dir


def load_weights(model: torch.nn.Module, args: argparse):
    """
    Load pretrained weights from given path.
    """
    pretrained_path = args.pretrained_path

    if not pretrained_path:
        return model

    print(f'Loading weights from {pretrained_path}')

    # state_dict = torch.load(pretrained_path)
    # sd_keys = list(state_dict.keys())

    checkpoint = torch.load(pretrained_path)
    state_dict = checkpoint['model_state_dict']
    sd_keys = list(state_dict.keys())
    if 'state' in sd_keys:
        state_dict = state_dict['state']
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                state_dict["{}".format(k[len('module.'):])] = state_dict[k]
                del state_dict[k]

        model.load_state_dict(state_dict, strict=False)

    elif 'params' in sd_keys:
        state_dict = state_dict['params']
        for k in list(state_dict.keys()):
            if k.startswith('encoder.'):
                state_dict["{}".format(k[len('encoder.'):])] = state_dict[k]

            del state_dict[k]

        model.load_state_dict(state_dict, strict=True)

    else:
        model.load_state_dict(state_dict)

    print("Weights loaded successfully ")
    return model


def get_fs_labels(method: str, num_way: int, num_query: int, num_shot: int):
    """
    Prepare few-shot labels. For example for 5-way, 1-shot, 2-query: [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, ...]
    """
    n_samples = num_shot + num_query if 'map' in method else num_query
    labels = torch.arange(num_way, dtype=torch.int16).repeat(n_samples).type(torch.LongTensor)

    if torch.cuda.is_available():
        return labels.cuda()
    else:
        return labels


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def print_and_log(results: dict, n: int = 0, logger: wandb = None):
    """
    Print and log current results.
    """
    for key in results.keys():
        # average by n if needed (n > 0)
        if n > 0 and 'time' not in key and '/epoch' not in key:
            results[key] = results[key] / n

        # print and log
        print(f'{key}: {results[key]:.4f}')

    if logger is not None:
        logger.log(results)


def get_t_sne(loader):
    data, label, _ = next(iter(loader))

    data = (data.reshape(data.shape[0], data.shape[1]*data.shape[2]*data.shape[3]))

    tsne = TSNE(n_components=2, n_jobs=-1)
    X_tsne = tsne.fit_transform(data)

    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=label)
    fig.update_layout(
        title="t-SNE Visualization of BreakHis-40x",
    )
    fig.show()

def set_seed(seed: int):
    """
    seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)



def clustering_accuracy(true_row_labels, predicted_row_labels):
    """
    The :mod:`coclust.evaluation.external` module provides functions
    to evaluate clustering or co-clustering results with external information
    such as the true labeling of the clusters.
    """

    """Get the best accuracy.
    Parameters
    ----------
    true_row_labels: array-like
        The true row labels, given as external information
    predicted_row_labels: array-like
        The row labels predicted by the model
    Returns
    -------
    float
        Best value of accuracy
    """

    cm = confusion_matrix(true_row_labels, predicted_row_labels)
    rows, cols = linear_assignment(_make_cost_m(cm))
    total = 0
    for row, column in zip(rows, cols):
        value = cm[row][column]
        total += value

    return (total * 1. / np.sum(cm)), cols


def _make_cost_m(cm):
    s = np.max(cm)
    return - cm + s

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class sla_pred_model(torch.nn.Module):
    def __init__(self, model, classifiers):
            super(sla_pred_model, self).__init__()
            self.base_model = model
            self.classifiers = torch.nn.ModuleList(classifiers)

    def forward(self, x, idx=0):
        features = self.base_model(x)
        if type(idx) is int:
            return self.classifiers[idx](features)
        else:
            if idx is None:
                idx = list(range(len(self.classifiers)))
            return [self.classifiers[i](features) for i in idx]


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'mem: {memory:.0f} '
                'mem reserved: {memory_res:.0f} '
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            len_iterable = len(iterable)
            if i % print_freq == 0 or i == len_iterable - 1:
                eta_seconds = iter_time.global_avg * (len_iterable - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.memory_allocated() / MB,
                        memory_res=torch.cuda.memory_reserved() / MB))
                else:
                    print(log_msg.format(
                        i, len_iterable, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))