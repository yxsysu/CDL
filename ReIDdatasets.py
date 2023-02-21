import h5py
import numpy as np
import torch.utils.data as data
import torch
import os
from PIL import Image
from scipy.io import savemat
import matplotlib.pyplot as plt
import re


def val_loader(path):
    img = Image.open(path).convert('RGB')
    return img


def _pluck_msmt(list_file, subdir, pattern=re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')):
    with open(list_file, 'r') as f:
        lines = f.readlines()

    ret = []
    pids = []
    for line in lines:
        line = line.strip()
        fname = line.split(' ')[0]
        pid, _, cam = map(int, pattern.search(os.path.basename(fname)).groups())
        if pid not in pids:
            pids.append(pid)
        ret.append((os.path.join(subdir, fname), pid, cam))

    return ret, pids


class MSMT(data.Dataset):
    def __init__(self, root, transform=None,
                 require_views=True,
                 state='train'):
        super(MSMT, self).__init__()
        self.root = root
        self.transform = transform
        self.require_views = require_views
        self.state = state
        self.list_file = os.path.join(self.root, 'list_'+self.state + '.txt')

        self.image_loader = val_loader
        if self.state == 'query' or self.state == 'gallery':
            sub_dir = 'test'
        else:
            sub_dir = 'train'
        self.ret, self.pid = _pluck_msmt(self.list_file, sub_dir)


    def turn_on_transform(self, transform=None):
        if transform is not None:
            self.transform = transform
        assert self.transform is not None, 'Transform not specified.'

    def __len__(self):
        return len(self.ret)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.ret[index][0])
        labels = self.ret[index][1]
        views = self.ret[index][2]
        image = self.image_loader(image_path)
        if self.transform is not None:
            image = self.transform(image)

        return image, labels, views, index


def get_view(filename, pattern=re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')):
    pid, _, cam = map(int, pattern.search(os.path.basename(filename)).groups())
    return cam


def get_id(filename, pattern=re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')):
    pid, _, cam = map(int, pattern.search(os.path.basename(filename)).groups())
    return pid


def extract_fname(line):
    line = line.strip()
    fname = os.path.join('train', line.split(' ')[0])
    return fname


def my_collect_fn(batch):
    images = []
    labels = []
    views = []
    for i in batch:
        images.append(i[0])
        labels.append(i[1])
        views.append(i[2])
    images = torch.cat(images)
    labels = torch.cat(labels)
    views = torch.cat(views)
    return images, labels, views


class PKTrackletMSMT(data.Dataset):
    def __init__(self, root, transform=None, require_views=True,
                 num_instance=2):
        super(PKTrackletMSMT, self).__init__()
        self.num_instance = num_instance
        self.root = root
        self.transform = transform
        self.require_views = require_views
        if self.transform is not None:
            self.on_transform = True
        else:
            self.on_transform = False
        self.image_loader = val_loader
        self.data = dict()
        self.mapping = dict()
        self.re_mapping = dict()

        list_file = os.path.join(self.root, 'list_train.txt')
        with open(list_file, 'r') as f:
            lines = f.readlines()

        with open(os.path.join(self.root, 'list_val.txt'), 'r') as f:
            new_lines = f.readlines()
        lines.extend(new_lines)

        all_images = np.array(list(map(extract_fname, lines)))

        views = np.array(list(map(get_view, all_images))) - 1
        labels = np.array(list(map(get_id, all_images)))
        unique_view = np.unique(views)
        self.camera_num = len(unique_view)
        self.image_num = dict()
        max_pid = max(labels)
        v_pid = np.zeros((self.camera_num, max_pid+1))

        self.labels_in_view = dict()
        self.tracklets_path = dict()
        self.data_label = dict()
        for view in range(self.camera_num):
            self.data[view] = all_images[views == view]
            self.image_num[view] = len(self.data[view])
            self.data_label[view] = labels[views == view]

            self.labels_in_view[view] = labels[views == view]
            temp_labels_in_views = self.labels_in_view[view]

            self.labels_in_view[view] = np.unique(self.labels_in_view[view])
            self.labels_in_view[view].sort()
            self.mapping[view] = dict(zip(self.labels_in_view[view], np.arange(len(self.labels_in_view[view]))))
            self.re_mapping[view] = dict(zip(np.arange(len(self.labels_in_view[view])), self.labels_in_view[view]))
            v_pid[view][self.labels_in_view[view]] = 1

            self.tracklets_path[view] = []
            for pid in range(len(self.labels_in_view[view])):
                self.tracklets_path[view].append([self.data[view][temp_labels_in_views == self.labels_in_view[view][pid]]])

        self.appear_in_view = dict()
        for view in range(self.camera_num):
            self.appear_in_view[view] = np.zeros((self.camera_num, len(self.labels_in_view[view])))
            for view_2 in range(self.camera_num):
                self.appear_in_view[view][view_2] = v_pid[view_2][self.labels_in_view[view]]
            self.appear_in_view[view] = torch.LongTensor(self.appear_in_view[view]).cuda()

        for view in range(self.camera_num):
            self.labels_in_view[view] = torch.LongTensor(self.labels_in_view[view]).cuda()
        self.num_class = self.return_num_class()
        self.shuffle_images()

    def return_num_class(self):
        t = []
        for i in range(self.camera_num):
            t.append(len(self.mapping[i]))
        return t

    def return_views(self):
        return self.camera_num

    def turn_on_transform(self, transform=None):
        self.on_transform = True
        if transform is not None:
            self.transform = transform
        assert self.transform is not None, 'Transform not specified.'

    def turn_off_transform(self):
        self.on_transform = False

    def __len__(self):
        return sum(self.num_class)

    def get_path(self, filename):
        return os.path.join(self.root, filename)

    def get_images(self, image_path):
        image = []
        for i in image_path:
            new_image = self.image_loader(i)
            if self.on_transform:
                new_image = self.transform(new_image)
            image.append(new_image)
        image = torch.stack(image)
        return image

    def shuffle_images(self):
        for i in range(self.camera_num):
            np.random.shuffle(self.tracklets_path[i])

    def return_appear_in_view(self):
        return self.appear_in_view

    def return_re_mapping(self):
        return self.re_mapping

    def return_labels_in_v(self):
        return self.labels_in_view

    def __getitem__(self, index):
        file_name = []
        labels = []
        views = []
        true_label = []

        view = 0
        index_in_view = index - 1
        for i in range(1, self.camera_num):
            if index > sum(self.num_class[:self.camera_num - i]):
                view = self.camera_num - i
                index_in_view = index - sum(self.num_class[:self.camera_num - i]) - 1
                break

        replace = False
        if len(self.tracklets_path[view][index_in_view]) < self.num_instance:
            replace = True
        t = np.random.choice(self.tracklets_path[view][index_in_view][0],
                             size=self.num_instance, replace=replace)
        file_name.extend(t)
        label_ = [int(i.split('/')[1]) for i in t]

        true_label.extend(label_)
        label = [self.mapping[view][i] for i in label_]
        labels.extend(label)
        views.extend([view]*self.num_instance)

        image_path = np.array(list(map(self.get_path, file_name)))
        images = self.get_images(image_path)
        labels = torch.LongTensor(labels)
        views = torch.LongTensor(views)
        return images, labels, views, true_label


class TrackletMSMT(data.Dataset):
    def __init__(self, root, transform=None, require_views=True,
                 num_instance=2):
        super(TrackletMSMT, self).__init__()
        self.num_instance = num_instance
        self.root = root
        self.transform = transform
        self.require_views = require_views
        if self.transform is not None:
            self.on_transform = True
        else:
            self.on_transform = False
        self.image_loader = val_loader
        self.data = dict()
        self.mapping = dict()
        self.re_mapping = dict()

        list_file = os.path.join(self.root, 'list_train.txt')
        with open(list_file, 'r') as f:
            lines = f.readlines()

        with open(os.path.join(self.root, 'list_val.txt'), 'r') as f:
            new_lines = f.readlines()
        lines.extend(new_lines)

        all_images = np.array(list(map(extract_fname, lines)))

        views = np.array(list(map(get_view, all_images))) - 1
        labels = np.array(list(map(get_id, all_images)))
        unique_view = np.unique(views)
        self.camera_num = len(unique_view)
        self.image_num = dict()
        max_pid = max(labels)
        v_pid = np.zeros((self.camera_num, max_pid+1))

        self.labels_in_view = dict()
        self.tracklets_path = dict()
        self.data_label = dict()
        for view in range(self.camera_num):
            self.data[view] = all_images[views == view]
            self.image_num[view] = len(self.data[view])
            self.data_label[view] = labels[views == view]

            self.labels_in_view[view] = labels[views == view]
            temp_labels_in_views = self.labels_in_view[view]

            self.labels_in_view[view] = np.unique(self.labels_in_view[view])
            self.labels_in_view[view].sort()
            self.mapping[view] = dict(zip(self.labels_in_view[view], np.arange(len(self.labels_in_view[view]))))
            self.re_mapping[view] = dict(zip(np.arange(len(self.labels_in_view[view])), self.labels_in_view[view]))
            v_pid[view][self.labels_in_view[view]] = 1

            self.tracklets_path[view] = []
            for pid in range(len(self.labels_in_view[view])):
                self.tracklets_path[view].append([self.data[view][temp_labels_in_views == self.labels_in_view[view][pid]]])

        self.appear_in_view = dict()
        for view in range(self.camera_num):
            self.appear_in_view[view] = np.zeros((self.camera_num, len(self.labels_in_view[view])))
            for view_2 in range(self.camera_num):
                self.appear_in_view[view][view_2] = v_pid[view_2][self.labels_in_view[view]]
            self.appear_in_view[view] = torch.LongTensor(self.appear_in_view[view]).cuda()

        for view in range(self.camera_num):
            self.labels_in_view[view] = torch.LongTensor(self.labels_in_view[view]).cuda()
        self.num_class = self.return_num_class()

    def return_num_class(self):
        t = []
        for i in range(self.camera_num):
            t.append(len(self.mapping[i]))
        return t

    def return_views(self):
        return self.camera_num

    def turn_on_transform(self, transform=None):
        self.on_transform = True
        if transform is not None:
            self.transform = transform
        assert self.transform is not None, 'Transform not specified.'

    def turn_off_transform(self):
        self.on_transform = False

    def __len__(self):
        return sum(self.num_class)

    def get_path(self, filename):
        return os.path.join(self.root, filename)

    def get_images(self, image_path):
        image = []
        for i in image_path:
            new_image = self.image_loader(i)
            if self.on_transform:
                new_image = self.transform(new_image)
            image.append(new_image)
        image = torch.stack(image)
        return image

    def shuffle_images(self):
        for i in range(self.camera_num):
            np.random.shuffle(self.tracklets_path[i])

    def return_appear_in_view(self):
        return self.appear_in_view

    def return_re_mapping(self):
        return self.re_mapping

    def return_labels_in_v(self):
        return self.labels_in_view

    def __getitem__(self, index):
        file_name = []
        labels = []
        views = []
        true_label = []

        view = 0
        index_in_view = index - 1
        for i in range(1, self.camera_num):
            if index > sum(self.num_class[:self.camera_num - i]):
                view = self.camera_num - i
                index_in_view = index - sum(self.num_class[:self.camera_num - i]) - 1
                break

        t = self.tracklets_path[view][index_in_view][0]
        file_name.extend(t)

        label_ = [int(i.split('/')[1]) for i in t]

        # trackletMSMT

        true_label.extend(label_)
        label = [self.mapping[view][i] for i in label_]
        labels.extend(label)
        views.extend([view]*len(t))

        image_path = np.array(list(map(self.get_path, file_name)))
        images = self.get_images(image_path)
        labels = torch.LongTensor(labels)
        views = torch.LongTensor(views)
        return images, labels, views, true_label


def main():
    debug = 1

if __name__ == '__main__':
    main()
