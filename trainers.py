from resnet_bn_normface import *
from utils import *
import torch.nn as nn
import torch
import os
import torch.nn.functional as F
from optimizer import build_optimizer
from torch.optim.lr_scheduler import MultiStepLR
from custom_loss import CrossEntropyLoss, \
    IDP, AlignLoss


class Trainer(object):
    def __init__(self):
        super(Trainer, self).__init__()

    def train(self, *names):
        """
        set the given attributes in names to the training state.
        if names is empty, call the train() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).train()

    def eval(self, *names):
        """
        set the given attributes in names to the evaluation state.
        if names is empty, call the eval() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).eval()


class ReidTrainer(Trainer):
    def __init__(self, cfg, num_classes, pretrained_state_dict=None,
                 strict=False):
        super(ReidTrainer, self).__init__()
        self.cfg = cfg
        self.scale = 1.0
        self.ce_loss = CrossEntropyLoss(cfg.loss.epsilon)
        if not cfg.loss.align_loss_v2:
            self.align_loss = AlignLoss(batch_size=int(sum(num_classes)*cfg.loss.p),
                                        p=cfg.loss.p,
                                        less_n_pid=cfg.train.less_n_pid)
            print('using v1 align loss')
        else:
            raise NotImplementedError

        if cfg.net.normface:
            print('using resnet50 + bn + normface')
            self.net = resnet50_bn_normface(pretrained=False, num_classes=num_classes,
                                            relu=cfg.net.relu, bias=cfg.net.bias,
                                            stride=cfg.net.stride)
            self.net_ema = resnet50_bn_normface(pretrained=False, num_classes=num_classes,
                                                relu=cfg.net.relu, bias=cfg.net.bias,
                                                stride=cfg.net.stride)
            self.scale = cfg.net.scale
        else:
            raise NotImplementedError

        if pretrained_state_dict is not None:
            self.net.load_state_dict(pretrained_state_dict, strict=strict)
            self.net_ema.load_state_dict(pretrained_state_dict, strict=strict)
        self.net = torch.nn.DataParallel(self.net).to(torch.device('cuda'))
        self.net_ema = torch.nn.DataParallel(self.net_ema).to(torch.device('cuda'))
        for param in self.net_ema.parameters():
            param.detach_()

        self.views = len(num_classes)
        self.net = self.net.cuda()
        if not cfg.loss.align_loss_v2:
            self.align_loss.init_center([i.weight for i in self.net.module.fc])
        bn_params, other_params, fc_params, embed_params = partition_params(self.net, 'bn')
        param_groups = [{'params': bn_params, 'weight_decay': 0},
                        {'params': other_params},
                        {'params': embed_params, 'lr': cfg.train.em_lr},
                        {'params': fc_params, 'lr': cfg.train.fc_lr}]
        self.optimizer = build_optimizer(param_groups, optim=self.cfg.optimizer,
                                         lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

        if len(cfg.optimizer_step) == 0:
            milestones = [int(cfg.train.max_epoch / 8 * 3),
                          int(cfg.train.max_epoch / 8 * 5),
                          int(cfg.train.max_epoch / 8 * 7)]
        else:
            milestones = cfg.optimizer_step
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=milestones, gamma=self.cfg.optimizer_gamma)

        self.IDPLoss = IDP(views=self.views,
                           scale=self.scale / cfg.loss.tau,
                           sim_threshold=cfg.loss.filter.sim_threshold,
                           attention=cfg.loss.filter.enable,
                           momentum=cfg.loss.filter.momentum,
                           renorm=cfg.loss.filter.renorm,
                           renorm_scale=cfg.loss.filter.renorm_scale,
                           less_n_pid=cfg.train.less_n_pid,
                           tau2=cfg.loss.tau2,
                           mutual=cfg.loss.filter.mutual)
        self.tau2 = cfg.loss.tau2
        print("tau2: ", self.tau2)
        self.IDPLoss.init_view_classifier(self.net.module.return_agents())
        self.IDPLoss.init_updated_classifier()

        self.view_split = dict()
        self.view_split[0] = [0, num_classes[0]]
        for i in range(1, self.views):
            self.view_split[i] = [sum(num_classes[:i]), sum(num_classes[:i+1])]

    def update_momentum(self, new_momentum):
        self.IDPLoss.set_momentum(new_momentum)

    def update_IDP_threshold(self, step_size):
        sim_threshold = self.IDPLoss.sim_threshold
        sim_threshold = sim_threshold + step_size
        self.IDPLoss.set_threshold(sim_threshold)

    def train_epoch(self, loader, epoch, views,
                    lamda=0.0, RKA=0.0):
        batch_time_meter = AverageMeter()
        stats = ('intra_loss', 'idp_loss', 'total_loss', 'align_loss')

        meters_trn = {stat: AverageMeter() for stat in stats}
        self.net.train()
        self.net_ema.train()

        end = time.time()
        # self.RKLLoss.reset_record_dict()

        num_step = len(loader)

        for i, tuple in enumerate(loader):

            split_one = torch.ones_like(tuple[2])
            mask = torch.arange(1, len(tuple[2]), 2)
            split_one[mask] = 0
            mask = split_one

            imgs = torch.cat((tuple[0][mask.eq(1)], tuple[0][mask.eq(0)]))
            labels = torch.cat((tuple[1][mask.eq(1)], tuple[1][mask.eq(0)]))
            views = torch.cat((tuple[2][mask.eq(1)], tuple[2][mask.eq(0)]))
            imgs = imgs.to(torch.device('cuda'))
            labels = labels.to(torch.device('cuda'))
            views = views.to(torch.device('cuda'))

            similarity, order_feature = self.net(imgs, views)
            _ = self.net_ema(imgs, views)

            agents = self.net.module.return_agents_grad()
            self.optimizer.zero_grad()
            intra_loss = []
            center_loss = []
            v_labels = []
            v_true_labels = []
            for v in torch.unique(views):
                labels_in_v = labels[views.eq(v)]
                v_labels.append(labels_in_v)

                sim = similarity[views.eq(v)][:, self.view_split[v.item()][0]:self.view_split[v.item()][1]]
                sim = sim * self.tau2

                intra_loss.append(F.cross_entropy(sim, labels_in_v))


            intra_loss = torch.stack(intra_loss)
            intra_loss = intra_loss.mean()

            idp_loss = torch.tensor([0.0]).cuda()
            if lamda > 0:
                ema_agents = self.net_ema.module.return_agents_grad()
                idp_loss = self.IDPLoss(similarity, agents, self.view_split,
                                        views, ema_agents)
                idp_loss = lamda * idp_loss

            align_loss = self.align_loss(agents) * RKA
            total_loss = intra_loss + idp_loss + align_loss

            total_loss.backward()
            self.optimizer.step()

            self._update_ema_variables(self.net, self.net_ema, 0.999, epoch * len(loader) + i)

            for k in stats:
                v = locals()[k]
                if v.item() > 0:
                    meters_trn[k].update(v.item(), self.cfg.train.batch_size)

            batch_time_meter.update(time.time() - end)
            freq = self.cfg.train.batch_size / batch_time_meter.avg
            end = time.time()
            if self.cfg.print_freq != 0 and i % self.cfg.print_freq == 0:
                list_lr = self.lr_scheduler.get_lr()
                print('  Iter: [{:03d}/{:03d}] lr: [{:2e}, {:.2e}, {:.2e}, {:.2e}] Freq {:.1f}   '.
                      format(i, len(loader), list_lr[0], list_lr[1], list_lr[2], list_lr[3],
                             freq) + create_stat_string(meters_trn) + time_string())

        return meters_trn

    def eval_performance(self, gallery_loader, probe_loader):
        ranks = [1, 5, 10, 20]
        self.eval()

        gallery_features, gallery_labels, gallery_views = extract_features(gallery_loader, self.net_ema,
                                                                           index_feature=0, return_numpy=False,
                                                                           loop=2)
        probe_features, probe_labels, probe_views = extract_features(probe_loader, self.net_ema,
                                                                     index_feature=0, return_numpy=False,
                                                                     loop=2)
        gallery_features = gallery_features.cpu()
        probe_features = probe_features.cpu()
        dist = 1 - gallery_features.mm(probe_features.t())
        dist = dist.cpu()
        CMC, MAP = eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views, probe_views, ignore_MAP=False)
        del dist
        print('** Results **')
        print('mAP: {:.1%}'.format(MAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, CMC[r - 1]))

        return CMC[0]

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def save_checkpoint(self, name='model.pth'):
        save_dir = os.path.join(self.cfg.data.save_dir, name)
        view_classifier = []
        for i in self.IDPLoss.view_classifier:
            temp_i = []
            for j in i:
                temp_i.append(j.cpu())
            view_classifier.append(temp_i)
        state = {'net': self.net.state_dict(),
                 'view_classifier': view_classifier,
                 'ema_net': self.net_ema.state_dict()}
        torch.save(state, save_dir)

