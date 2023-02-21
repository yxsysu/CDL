import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import AverageMeter
import math
import copy
import numpy as np


class CrossEntropyLoss(nn.Module):
    r"""Cross entropy loss with label smoothing regularizer.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by

    .. math::
        \begin{equation}
        (1 - \epsilon) \times y + \frac{\epsilon}{K},
        \end{equation}

    where :math:`K` denotes the number of classes and :math:`\epsilon` is a weight. When
    :math:`\epsilon = 0`, the loss function reduces to the normal cross entropy.

    Args:
        num_classes (int): number of classes.
        epsilon (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, epsilon=0.1, use_gpu=True, label_smooth=True):
        super(CrossEntropyLoss, self).__init__()
        self.epsilon = epsilon if label_smooth else 0
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        log_probs = self.logsoftmax(inputs)
        #targets = torch.zeros(log_probs.size()).scatter(1, targets.unsqueeze(1).data.cpu(), 1)
        tar = torch.zeros_like(log_probs).to(torch.device('cuda'))
        targets = tar.scatter(1, targets.unsqueeze(1), 1)
        num_classes = targets.shape[1]
        targets = targets.to(torch.device('cuda'))
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        return (- targets * log_probs).mean(0).sum()


def pairwise_kl_div(x, y):
    """
    :param x:
    :param y:
    :return: x*log(x/y)
    """
    logit = torch.log(x.unsqueeze(1) / y.unsqueeze(0))
    x = x.unsqueeze(1).transpose(1, 2)
    kl_div = logit.bmm(x)
    return kl_div


def pairwise_kl_div_v2(logit_x, logit_y):
    """
    :param logit_x:
    :param logit_y:
    :return: p(x)*[log(p(x)) - log(p(y))]
    """
    log_px = F.log_softmax(logit_x, dim=1)
    log_py = F.log_softmax(logit_y, dim=1)
    px = F.softmax(logit_x, dim=1)

    px = px.unsqueeze(1).transpose(1, 2)
    logit = log_px.unsqueeze(1) - log_py.unsqueeze(0)
    kl_div = logit.bmm(px)
    return kl_div


class IDP(nn.Module):
    def __init__(self, views=6, scale=1,
                 sim_threshold=0.5,
                 momentum=1.0,
                 renorm=False,
                 renorm_scale=-1,
                 attention=False,
                 less_n_pid=100,
                 tau2=25.0,
                 mutual=True):
        super(IDP, self).__init__()
        self.mutual = mutual
        self.tau2 = tau2
        print('self.tau2: ', self.tau2)
        
        self.less_n_pid = less_n_pid

        self.views = views
        self.index = np.arange(self.views)
        self.distance = pairwise_kl_div_v2
        self.scale = scale
        print('the scale in loss is ', scale)

        self.sim_threshold = sim_threshold

        self.attention = attention
        if self.attention:
            print('using attention')
            print('sim threshold: ', sim_threshold)

        self.view_classifier = []
        self.updated_classifier = []
        self.momentum = momentum
        print('self.momentum: ', self.momentum)

        self.renorm = renorm
        self.renorm_scale = renorm_scale
        print('renorm: ', self.renorm)
        print('renorm scale: ', self.renorm_scale)

        if self.mutual:
            print('using mutual selection')

    def set_momentum(self, new_momentum):
        self.momentum = new_momentum
        print('now self.momentum: ', self.momentum)

    def init_view_classifier(self, agents):
        self.index = []
        for i in range(len(agents)):
            if len(agents[i]) >= self.less_n_pid:
                self.index.append(i)
        self.index = np.array(self.index)
        for i in range(self.views):
            result = []
            for j in range(self.views):
                sim = agents[i].mm(agents[j].t())
                sim = F.softmax(sim * self.tau2, dim=1).detach()
                result.append(sim)
            self.view_classifier.append(result)


    def init_updated_classifier(self):
        self.updated_classifier = []
        for i in self.view_classifier:
            temp_i = []
            for j in i:
                temp_i.append(j.clone())
            self.updated_classifier.append(temp_i)

    def transfer_classifier(self):
        self.view_classifier = []
        for i in self.updated_classifier:
            temp_i = []
            for j in i:
                temp_i.append(j.clone())
            self.view_classifier.append(temp_i)

    def attn_func(self, t_agent, s_agent,
                  t_ind, s_ind, update=True):
        sim = t_agent.mm(s_agent.t())
        sim = F.softmax(sim * self.tau2, dim=1).detach()

        if update:
            self.updated_classifier[t_ind][s_ind] = self.momentum * sim \
                                                    + (1 - self.momentum) * self.updated_classifier[t_ind][s_ind]
        appear, labels = torch.max(self.view_classifier[t_ind][s_ind], dim=1)
        appear = appear.ge(self.sim_threshold).detach()

        if self.mutual:
            re_appear, _ = torch.max(self.view_classifier[s_ind][t_ind][labels], dim=1)
            re_appear = re_appear.ge(self.sim_threshold).detach()
            appear = appear & re_appear

        return appear

    def set_threshold(self, sim_threshold):
        self.sim_threshold = sim_threshold
        print('now sim threshold in IDP: ', sim_threshold)

    def preservation(self, sim, agents, t_ind, s_ind, view_split,
                     views, ema_agents, update=True):
        t_sim = sim[views.eq(t_ind)][:, view_split[t_ind][0]:view_split[t_ind][1]] * self.scale
        if self.renorm:
            t_agents = agents[t_ind].detach()
            reweight = t_agents.mm(t_agents.t())
            if self.renorm_scale > 0:
                reweight = reweight * self.renorm_scale
                reweight = F.softmax(reweight, dim=1)
            t_sim = t_sim.mm(reweight)
        appear_in_s = self.attn_func(ema_agents[t_ind], ema_agents[s_ind],
                                     t_ind, s_ind, update)
        if appear_in_s.sum() <= 10:
            return None
        if self.attention:
            t_sim = t_sim[:, appear_in_s]
        if t_sim.shape[0] < 3:
            return None

        s_sim = sim[views.eq(t_ind)][:, view_split[s_ind][0]:view_split[s_ind][1]] * self.scale
        if self.renorm:
            s_agents = agents[s_ind].detach()
            reweight = s_agents.mm(s_agents.t())
            if self.renorm_scale > 0:
                reweight = reweight * self.renorm_scale
                reweight = F.softmax(reweight, dim=1)
            s_sim = s_sim.mm(reweight)
        if self.attention:
            appear_in_t = self.attn_func(ema_agents[s_ind], ema_agents[t_ind],
                                         s_ind, t_ind, update)
            if appear_in_t.sum() <= 0:
                return None
            s_sim = s_sim[:, appear_in_t]

        t_kl = self.distance(t_sim, t_sim).detach().view(-1)
        s_kl = self.distance(s_sim, s_sim).view(-1)
        part_loss = F.smooth_l1_loss(s_kl, t_kl, reduction='mean')
        return part_loss

    def forward(self, sim, agents, view_split, views, ema_agents):
        np.random.shuffle(self.index)
        loss = []

        index_s_ind = [i.item() for i in torch.unique(views)]
        for i in torch.unique(views):
            t_ind = i.item()
            for j in index_s_ind:
                if i == j:
                    continue
                s_ind = j
                part_loss = self.preservation(sim, agents, t_ind, s_ind,
                                              view_split, views, ema_agents)
                if part_loss is not None:
                    loss.append(part_loss)

        if len(loss) == 0:
            loss = torch.tensor([0.0]).cuda()
        else:
            loss = torch.mean(torch.stack(loss))
        return loss


class AlignLoss(torch.nn.Module):
    def __init__(self, batch_size,
                 p=1.0, less_n_pid=-1):
        super(AlignLoss, self).__init__()
        self.moment = batch_size / 10000
        self.initialized = False
        self.p = p
        self.less_n_pid = less_n_pid
        print('align loss ---')
        print('batch size: ', batch_size)
        print('moment: ', self.moment)

    def set_moment(self, moment):
        self.moment = moment
        print('now moment of align loss is ', self.moment)

    def init_center(self, agents):
        means = []
        stds = []
        for v in range(len(agents)):
            ml_in_v = agents[v].detach()
            ml_in_v = F.normalize(ml_in_v, dim=1)
            if len(ml_in_v) == 1:
                continue
            mean = ml_in_v.mean(dim=0)
            means.append(mean)
            std = ml_in_v.std(dim=0)
            stds.append(std)
        center_mean = torch.mean(torch.stack(means), dim=0)
        center_std = torch.mean(torch.stack(stds), dim=0)
        self.register_buffer('center_mean', center_mean)
        self.register_buffer('center_std', center_std)

    def _update_centers(self, agents):
        """
        :param variables: shape=(BS, n_class)
        :param views: shape=(BS,)
        :return:
        """

        means = []
        stds = []
        for v in range(len(agents)):
            ml_in_v = agents[v].detach()
            if len(ml_in_v) == 1:
                continue
            mean = ml_in_v.mean(dim=0)
            means.append(mean)
            std = ml_in_v.std(dim=0)
            stds.append(std)
        new_mean = torch.mean(torch.stack(means), dim=0)
        self.center_mean = self.center_mean*(1-self.moment) + new_mean*self.moment
        new_std = torch.mean(torch.stack(stds), dim=0)
        self.center_std = self.center_std*(1-self.moment) + new_std*self.moment

    def random_choice(self, agent):
        num = len(agent)
        index = np.arange(num)
        selected_numm = int(self.p * num)
        index = np.random.choice(index, selected_numm, replace=False)
        index = torch.LongTensor(index).cuda()
        return agent[index]

    def forward(self, t_agents):
        """
        :param t_agents: shape=[(BS, n_class)]
        :return:
        """
        agents = [self.random_choice(i) for i in t_agents if len(i) >= self.less_n_pid]

        self._update_centers(agents)

        loss_terms = []
        for v in range(len(agents)):
            ml_in_v = agents[v]
            if len(ml_in_v) == 1:
                continue
            mean = ml_in_v.mean(dim=0)
            loss_mean = (mean - self.center_mean).pow(2).sum()
            loss_terms.append(loss_mean)
            std = ml_in_v.std(dim=0)
            loss_std = (std - self.center_std).pow(2).sum()
            loss_terms.append(loss_std)
        loss_total = torch.mean(torch.stack(loss_terms))
        return loss_total


def main():
    x = torch.rand(size=(64, 2048)).clamp(min=0.1) * 5
    y = torch.rand(size=(64, 2048)).clamp(min=0.1) * 5
    px = F.softmax(x, dim=1)
    py = F.softmax(y, dim=1)
    t = []
    t_1 = []
    gtf = torch.nn.KLDivLoss(reduction='batchmean')



if __name__ == '__main__':
    main()
