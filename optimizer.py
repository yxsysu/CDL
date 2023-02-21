from __future__ import absolute_import
from __future__ import print_function

import warnings

import torch
import torch.nn as nn

AVAI_OPTIMS = ['adam', 'amsgrad', 'sgd', 'rmsprop']


def build_optimizer(
        param_groups,
        optim='adam',
        lr=0.0003,
        weight_decay=5e-04,
        momentum=0.9,
        sgd_dampening=0,
        sgd_nesterov=False,
        rmsprop_alpha=0.99,
        adam_beta1=0.9,
        adam_beta2=0.99,
):

    if optim == 'adam':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optim == 'amsgrad':
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    return optimizer