import os, sys, time
import numpy as np
import matplotlib
import subprocess
import torch
import random
import argparse
import torchvision.transforms as transforms
import torchvision
from random import sample
matplotlib.use('agg')
from ReIDdatasets import *
import torch.nn.functional as F
import torch.cuda as cutorch
import yaml
import errno
import os.path as osp
import math
from collections import OrderedDict


class BaseOptions(object):
    """
    base options for deep learning for Re-ID.
    parse basic arguments by parse(), print all the arguments by print_options()
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.args = None

        self.parser.add_argument('--save_path', type=str, default='debug', help='Folder to save checkpoints and log.')
        self.parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--gpu', type=str, default='0', help='gpu used.')

    def parse(self):
        self.args = self.parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        with open(os.path.join(self.args.save_path, 'args.yaml')) as f:
            extra_args = yaml.load(f)
        self.args = argparse.Namespace(**vars(self.args), **extra_args)
        return self.args

    def print_options(self, logger):
        logger.print_log("")
        logger.print_log("----- options -----".center(120, '-'))
        args = vars(self.args)
        string = ''
        for i, (k, v) in enumerate(sorted(args.items())):
            string += "{}: {}".format(k, v).center(40, ' ')
            if i % 3 == 2 or i == len(args.items())-1:
                logger.print_log(string)
                string = ''
        logger.print_log("".center(120, '-'))
        logger.print_log("")


def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger(object):
    """Writes console output to external text file.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_

    Args:
        fpath (str): directory to save logging file.

    Examples::
       >>> import sys
       >>> import os
       >>> import os.path as osp
       >>> from torchreid.utils import Logger
       >>> save_dir = 'log/resnet50-softmax-market1501'
       >>> log_name = 'train.log'
       >>> sys.stdout = Logger(osp.join(args.save_dir, log_name))
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def time_string():
    ISOTIMEFORMAT = '%Y-%m-%d %X'
    string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
    return string


def flip_tensor(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_features(loader, model, index_feature=None, return_numpy=True, loop=1):
    """
    extract features for the given loader using the given model
    if loader.dataset.require_views is False, the returned 'views' are empty.
    :param loader: a ReIDDataset that has attribute require_views
    :param model: returns a tuple containing the feature or only return the feature. if latter, index_feature be None
    model can also be a tuple of nn.Module, indicating that the feature extraction is multi-stage.
    in this case, index_feature should be a tuple of the same size.
    :param index_feature: in the tuple returned by model, the index of the feature.
    if the model only returns feature, this should be set to None.
    :param return_numpy: if True, return numpy array; otherwise return torch tensor
    :return: features, labels, views, np array
    """
    if type(model) is not tuple:
        models = (model,)
        indices_feature = (index_feature,)
    else:
        assert len(model) == len(index_feature)
        models = model
        indices_feature = index_feature
    for m in models:
        m.eval()

    labels = []
    views = []
    features = []

    require_views = loader.dataset.require_views
    for i, data in enumerate(loader):
        imgs = data[0].cuda()
        label_batch = data[1]
        inputs = imgs
        global_feature = None
        for i in range(loop):
            if i == 1:
                inputs = flip_tensor(imgs)
            for m, feat_idx in zip(models, indices_feature):
                with torch.no_grad():
                    output_tuple = m(inputs)
                feature_batch = output_tuple if feat_idx is None else output_tuple[feat_idx]
                inputs = feature_batch
            if global_feature is None:
                global_feature = feature_batch
            else:
                global_feature = global_feature + feature_batch
                global_feature = global_feature / 2
        features.append(F.normalize(global_feature))
        labels.append(label_batch)
        if require_views:
            view_batch = data[2]
            views.append(view_batch)
    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0)
    views = torch.cat(views, dim=0) if require_views else views
    if return_numpy:
        return np.array(features.cpu()), np.array(labels.cpu()), np.array(views.cpu())
    else:
        return features, labels, views


def create_stat_string(meters):
    stat_string = ''
    for stat, meter in meters.items():
        stat_string += '{} {:.3f}   '.format(stat, meter.avg)
    return stat_string


def convert_secs2time(epoch_time):
    need_hour = int(epoch_time / 3600)
    need_mins = int((epoch_time - 3600 * need_hour) / 60)
    need_secs = int(epoch_time - 3600 * need_hour - 60 * need_mins)
    return need_hour, need_mins, need_secs


def eval_cmc_map(dist, gallery_labels, probe_labels, gallery_views=None,
                 probe_views=None, ignore_MAP=True):
    """
    :param dist: 2-d np array, shape=(num_gallery, num_probe), distance matrix.
    :param gallery_labels: np array, shape=(num_gallery,)
    :param probe_labels:
    :param gallery_views: np array, shape=(num_gallery,) if specified, for any probe image,
    the gallery correct matches from the same view are ignored.
    :param probe_views: must be specified if gallery_views are specified.
    :param ignore_MAP: is True, only compute cmc
    :return:
    CMC: np array, shape=(num_gallery,). Measured by percentage
    MAP: np array, shape=(1,). Measured by percentage
    """
    gallery_labels = np.asarray(gallery_labels)
    probe_labels = np.asarray(probe_labels)
    dist = np.asarray(dist)

    is_view_sensitive = False
    num_gallery = gallery_labels.shape[0]
    num_probe = probe_labels.shape[0]
    if gallery_views is not None or probe_views is not None:
        assert gallery_views is not None and probe_views is not None, \
            'gallery_views and probe_views must be specified together. \n'
        gallery_views = np.asarray(gallery_views)
        probe_views = np.asarray(probe_views)
        is_view_sensitive = True
    cmc = np.zeros((num_gallery, num_probe))
    ap = np.zeros((num_probe,))
    for i in range(num_probe):
        cmc_ = np.zeros((num_gallery,))
        dist_ = dist[:, i]
        probe_label = probe_labels[i]
        gallery_labels_ = gallery_labels
        if is_view_sensitive:
            probe_view = probe_views[i]
            is_from_same_view = gallery_views == probe_view
            is_correct = gallery_labels == probe_label
            should_be_excluded = is_from_same_view & is_correct
            dist_ = dist_[~should_be_excluded]
            gallery_labels_ = gallery_labels_[~should_be_excluded]
        ranking_list = np.argsort(dist_)
        inference_list = gallery_labels_[ranking_list]
        positions_correct_tuple = np.nonzero(probe_label == inference_list)
        positions_correct = positions_correct_tuple[0]
        pos_first_correct = positions_correct[0]
        cmc_[pos_first_correct:] = 1
        cmc[:, i] = cmc_

        if not ignore_MAP:
            num_correct = positions_correct.shape[0]
            for j in range(num_correct):
                last_precision = float(j) / float(positions_correct[j]) if j != 0 else 1.0
                current_precision = float(j + 1) / float(positions_correct[j] + 1)
                ap[i] += (last_precision + current_precision) / 2.0 / float(num_correct)

    CMC = np.mean(cmc, axis=1)
    MAP = np.mean(ap)
    return CMC, MAP


def occupy_gpu_memory(gpu_id=0, maximum_usage=None, buffer_memory=1000, memory_to_occupy=None):
    """
    As pytorch is dynamic, you might wanna take enough GPU memory to avoid OOM when you run your code
    in a messy server.
    if maximum_usage is specified, this function will return a dummy buffer which takes memory of
    (current_available_memory - (maximum_usage - current_usage) - buffer_memory) MB.
    otherwise, maximum_usage would be replaced by maximum usage till now, which is returned by
    torch.cuda.max_memory_cached()
    :param gpu_id:
    :param maximum_usage: float, measured in MB
    :param buffer_memory: float, measured in MB
    :return:
    """
    gpu_id = int(gpu_id)
    if maximum_usage is None:
        maximum_usage = cutorch.max_memory_cached()
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'])
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split(b'\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    available_memory = gpu_memory_map[gpu_id]
    if available_memory < buffer_memory+1000:
        print('Gpu memory has been mostly occupied (although maybe not by you)!')
    else:
        if memory_to_occupy is None:
            memory_to_occupy = int((available_memory - (maximum_usage - cutorch.memory_cached()) - buffer_memory))
        dim = int(memory_to_occupy * 1024 * 1024 * 8 / 32)
        x = torch.zeros(dim, dtype=torch.int)
        x.pin_memory()
        print('Occupied {}MB extra gpu memory.'.format(memory_to_occupy))
        x_ = x.cuda()
        del x_


def compute_accuracy(predictions, labels):
    """
    compute classification accuracy, measured by percentage.
    :param predictions: tensor. size = N*d
    :param labels: tensor. size = N
    :return: python number, the computed accuracy
    """
    predicted_labels = torch.argmax(predictions, dim=1)
    n_correct = torch.sum(predicted_labels == labels).item()
    batch_size = torch.numel(labels)
    acc = float(n_correct) / float(batch_size)
    return acc * 100


def eval_acc(dist, gallery_labels, probe_labels):
    gallery_labels = np.asarray(gallery_labels)
    probe_labels = np.asarray(probe_labels)
    dist = np.asarray(dist)

    ranking_table = np.argsort(dist, axis=0)
    r1_idx = ranking_table[0]
    infered_labels = gallery_labels[r1_idx]
    acc = (infered_labels == probe_labels).mean()*100
    return acc


def partition_params(module, strategy, *desired_modules):
    """
    partition params into desired part and the residual
    :param module:
    :param strategy: choices are: ['bn', 'specified'].
    'bn': desired_params = bn_params
    'specified': desired_params = all params within desired_modules
    :param desired_modules: strings, each corresponds to a specific module
    :return: two lists
    """
    if strategy == 'bn':
        desired_params_set = set()
        for m in module.modules():
            if (isinstance(m, torch.nn.BatchNorm1d) or
                    isinstance(m, torch.nn.BatchNorm2d) or
                    isinstance(m, torch.nn.BatchNorm3d)):
                desired_params_set.update(set(m.parameters()))
    elif strategy == 'specified':
        desired_params_set = set()
        for module_name in desired_modules:
            sub_module = module.__getattr__(module_name)
            for m in sub_module.modules():
                desired_params_set.update(set(m.parameters()))
    else:
        assert False, 'unknown strategy: {}'.format(strategy)
    fc_params_set = set(module.module.fc.parameters())
    embed_params_set = set(module.module.embedding.parameters())
    all_params_set = set(module.module.parameters())
    other_params_set = all_params_set.difference(desired_params_set)
    other_params_set = other_params_set.difference(fc_params_set)
    other_params_set = other_params_set.difference(embed_params_set)
    desired_params_set = desired_params_set.difference(embed_params_set)
    desired_params = list(desired_params_set)
    other_params = list(other_params_set)
    fc_params = list(fc_params_set)
    embed_params = list(embed_params_set)
    return desired_params, other_params, fc_params, embed_params


class RandomErasing(object):
    """Randomly erases an image patch.

    Origin: `<https://github.com/zhunzhong07/Random-Erasing>`_

    Reference:
        Zhong et al. Random Erasing Data Augmentation.

    Args:
        probability (float, optional): probability that this operation takes place.
            Default is 0.5.
        sl (float, optional): min erasing area.
        sh (float, optional): max erasing area.
        r1 (float, optional): min aspect ratio.
        mean (list, optional): erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):
        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


def get_reid_dataloaders(img_size,
                         crop_size, padding, batch_size,
                         color_jitter=False,
                         random_erase=False,
                         sampler='random',
                         num_instance=2):
    """
    get train/gallery/probe dataloaders.
    :return:
    """
    gallery_data = MSMT('path/to/dataset', state='gallery')
    probe_data = MSMT('path/to/dataset', state='query')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_data = PKTrackletMSMT('path/to/dataset', num_instance=num_instance)

    assert sampler == 'PK'

    tr = [transforms.RandomHorizontalFlip(), transforms.Resize(img_size),
          transforms.RandomCrop(crop_size, padding)]
    if color_jitter:
        tr.append(transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0, hue=0))
        print('using color jitter')
    tr.extend([transforms.ToTensor(),
               transforms.Normalize(mean, std)])
    if random_erase:
        print('using random earsing')
        tr.append(RandomErasing())
    train_transform = transforms.Compose(tr)

    test_transform = transforms.Compose(
        [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize(mean, std)])

    train_data.turn_on_transform(transform=train_transform)
    gallery_data.turn_on_transform(transform=test_transform)
    probe_data.turn_on_transform(transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True, drop_last=False,
                                               collate_fn=my_collect_fn)
    gallery_loader = torch.utils.data.DataLoader(gallery_data, batch_size=256, shuffle=False,
                                                 num_workers=4, pin_memory=True)
    probe_loader = torch.utils.data.DataLoader(probe_data, batch_size=256, shuffle=False,
                                               num_workers=4, pin_memory=True)

    return train_loader, gallery_loader, probe_loader


def find_wrong_match(dist, gallery_labels, probe_labels, gallery_views=None, probe_views=None):
    """
    find the probe samples which result in a wrong match at rank-1.
    :param dist: 2-d np array, shape=(num_gallery, num_probe), distance matrix.
    :param gallery_labels: np array, shape=(num_gallery,)
    :param probe_labels:
    :param gallery_views: np array, shape=(num_gallery,) if specified, for any probe image,
    the gallery correct matches from the same view are ignored.
    :param probe_views: must be specified if gallery_views are specified.
    :return:
    prb_idx: list of int, length == n_found_wrong_prb
    gal_idx: list of np array, each of which associating with the element in prb_idx
    correct_indicators: list of np array corresponding to gal_idx, indicating whether that gal is a correct match.
    """
    is_view_sensitive = False
    num_probe = probe_labels.shape[0]
    if gallery_views is not None or probe_views is not None:
        assert gallery_views is not None and probe_views is not None, \
            'gallery_views and probe_views must be specified together. \n'
        is_view_sensitive = True
    prb_idx = []
    gal_idx = []
    correct_indicators = []

    for i in range(num_probe):
        dist_ = dist[:, i]
        probe_label = probe_labels[i]
        gallery_labels_ = gallery_labels
        if is_view_sensitive:
            probe_view = probe_views[i]
            is_from_same_view = gallery_views == probe_view
            is_correct = gallery_labels == probe_label
            should_be_excluded = is_from_same_view & is_correct
            dist_ = dist_[~should_be_excluded]
            gallery_labels_ = gallery_labels_[~should_be_excluded]
        ranking_list = np.argsort(dist_)
        inference_list = gallery_labels_[ranking_list]
        positions_correct_tuple = np.nonzero(probe_label == inference_list)
        positions_correct = positions_correct_tuple[0]
        pos_first_correct = positions_correct[0]
        if pos_first_correct != 0:
            prb_idx.append(i)
            gal_idx.append(ranking_list)
            correct_indicators.append(probe_label == inference_list)

    return prb_idx, gal_idx, correct_indicators


def plot_ranking_imgs(gal_dataset, prb_dataset, gal_idx, prb_idx, n_gal=8, size=(256, 128), save_path='',
                      correct_indicators=None, sample_prb=False, n_prb=8):
    """
    plot ranking imgs and save it.
    :param gal_dataset: should support indexing and return a tuple, in which the first element is an img,
           represented as np array
    :param prb_dataset:
    :param gal_idx: list of np.array, each of which corresponds to the element in prb_idx
    :param prb_idx: list of int, indexing the prb_dataset
    :param n_gal: number of gallery imgs shown in a row (for a probe).
    :param size: resize all shown imgs
    :param save_path: directory to save; the file name is ranking_(time string).png
    :param correct_indicators: list of np array corresponding to gal_idx, indicating whether that
           gal is a correct match. if specified, each correct match will has a small green box in the upper-left.
    :param sample_prb: if True, the prb_idx is randomly sampled n_prb samples; otherwise, keep the order of prb_idx
    and plot all the images specified in prb_idx.
    :param n_prb: if sample_prb is True, we sample n_prb probe images.
    :return:
    """
    assert len(prb_idx) == len(gal_idx)
    if correct_indicators is not None:
        assert len(prb_idx) == len(correct_indicators)
    box_size = tuple(map(lambda x: int(x/12.0), size))

    is_gal_on = gal_dataset.on_transform
    is_prb_on = prb_dataset.on_transform
    gal_dataset.turn_off_transform()
    prb_dataset.turn_off_transform()

    transform = transforms.Compose([transforms.Resize(size), transforms.ToTensor()])

    n_prb = len(prb_idx) if n_prb > len(prb_idx) else n_prb
    if correct_indicators is None:
        if sample_prb:
            used = sample(list(zip(prb_idx, gal_idx)), n_prb)
        else:
            used = list(zip(prb_idx, gal_idx))
        imgs = []
        for p_idx, g_idx_array in used:
            prb_img = transform(prb_dataset[p_idx][0])
            imgs.append(prb_img)
            n_gal_used = min(n_gal, len(g_idx_array))
            for g_idx in g_idx_array[:n_gal_used]:
                gal_img = transform(gal_dataset[g_idx][0])
                imgs.append(gal_img)
            for i in range(n_gal - n_gal_used):
                imgs.append(np.zeros_like(prb_img))
    else:
        if sample_prb:
            used = sample(list(zip(prb_idx, gal_idx, correct_indicators)), n_prb)
        else:
            used = list(zip(prb_idx, gal_idx, correct_indicators))
        imgs = []
        for p_idx, g_idx_array, correct_ind in used:
            prb_img = transform(prb_dataset[p_idx][0])
            imgs.append(prb_img)
            n_gal_used = min(n_gal, len(g_idx_array))
            for g_idx, is_correct_match in zip(g_idx_array[:n_gal_used], correct_ind[:n_gal_used]):
                gal_img = transform(gal_dataset[g_idx][0])
                if is_correct_match:
                    gal_img[0, :box_size[0], :box_size[1]].zero_()
                    gal_img[1, :box_size[0], :box_size[1]].fill_(1.0)
                    gal_img[2, :box_size[0], :box_size[1]].zero_()
                else:
                    gal_img[0, :box_size[0], :box_size[1]].fill_(1.0)
                    gal_img[1, :box_size[0], :box_size[1]].zero_()
                    gal_img[2, :box_size[0], :box_size[1]].zero_()
                imgs.append(gal_img)
            for i in range(n_gal - n_gal_used):
                imgs.append(np.zeros_like(prb_img))

    filename = os.path.join(save_path, 'ranking_{}.png'.format(time_string()))
    torchvision.utils.save_image(imgs, filename, nrow=n_gal+1)
    print('saved ranking images into {}'.format(filename))
    gal_dataset.on_transform = is_gal_on
    prb_dataset.on_transform = is_prb_on


def init_pretrained_weights(model, pretrain_dict):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def pair_idx_to_dist_idx(d, i, j):
    """
    :param d: numer of elements
    :param i: np.array. i < j in every element
    :param j: np.array
    :return:
    """
    assert np.sum(i < j) == len(i)
    index = d*i - i*(i+1)/2 + j - 1 - i
    return index.astype(int)


def open_specified_layers(model, open_layers):
    r"""Opens specified layers in model for training while keeping
    other layers frozen.

    Args:
        model (nn.Module): neural net model.
        open_layers (str or list): layers open for training.

    Examples::
        >>> from torchreid.utils import open_specified_layers
        >>> # Only model.classifier will be updated.
        >>> open_layers = 'classifier'
        >>> open_specified_layers(model, open_layers)
        >>> # Only model.fc and model.classifier will be updated.
        >>> open_layers = ['fc', 'classifier']
        >>> open_specified_layers(model, open_layers)
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    if isinstance(open_layers, str):
        open_layers = [open_layers]

    for layer in open_layers:
        assert hasattr(model, layer), '"{}" is not an attribute of the model, please provide the correct name'.format(layer)

    for name, module in model.named_children():
        if name in open_layers:
            print("{} is opened".format(name))
            module.train()
            for p in module.parameters():
                p.requires_grad = True
        else:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False


def test():
    labels = torch.tensor([1, 2, 3, 4, 5, 6]).cuda()
    pass


if __name__ == '__main__':
    test()
