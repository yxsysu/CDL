import argparse
from yacs.config import CfgNode as CN


def get_default_config():
    cfg = CN()
    cfg.print_freq = 128
    cfg.optimizer = 'sgd'
    cfg.optimizer_step = []
    cfg.optimizer_gamma = 0.1

    cfg.net = CN()
    cfg.net.bn = False
    cfg.net.normface = True
    cfg.net.relu = False
    cfg.net.bias = True
    cfg.net.stride = 1
    cfg.net.scale = 25.0

    # train
    cfg.train = CN()
    cfg.train.warm_up_epoch = -1
    cfg.train.load_weight = "resnet"
    cfg.train.lr = 3.5e-4
    cfg.train.fc_lr = 7e-4
    cfg.train.em_lr = 7e-4
    cfg.train.weight_decay = 5e-4
    cfg.train.max_epoch = 40
    cfg.train.batch_size = 2
    cfg.train.seed = 1  # random seed
    cfg.train.less_n_pid = 100
    cfg.train.save = False

    cfg.val = CN()
    cfg.val.freq = 8

    # aug
    cfg.aug = CN()
    cfg.aug.color_jitter = False
    cfg.aug.random_erase = False

    # loss
    cfg.loss = CN()
    cfg.loss.epsilon = 0.1
    cfg.loss.lamda = 0.0
    cfg.loss.RKA = 1.0

    cfg.loss.p = 1.0
    cfg.loss.align_loss_v2 = False
    cfg.loss.v2_p = 16
    cfg.loss.tau = 1.0
    cfg.loss.tau2 = 25.0
    cfg.loss.detach_feature = False

    cfg.loss.filter = CN()
    cfg.loss.filter.sim_threshold = 0.5
    cfg.loss.filter.momentum = 1.0
    cfg.loss.filter.update_epoch = [-1]
    cfg.loss.filter.step_size = 0.0
    cfg.loss.filter.renorm = True
    cfg.loss.filter.renorm_scale = 12.5
    cfg.loss.filter.view_update_freq = 1
    cfg.loss.filter.mutual = False
    cfg.loss.filter.enable = True

    # data

    cfg.data = CN()
    cfg.data.save_dir = 'log/debug'
    cfg.data.image_size = [256, 128]
    cfg.data.crop_size = [256, 128]
    cfg.data.padding = 7
    cfg.data.num_instance = 2
    cfg.data.sampler = 'PK'

    return cfg


