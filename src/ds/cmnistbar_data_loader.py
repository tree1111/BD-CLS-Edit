import os

import numpy as np
from PIL import Image

from torchvision.utils import save_image

import torch
from torchvision import transforms
from torchvision import datasets

import pandas as pd

def gen_scm(cg, n):
    if cg == 'full-ncm':
        # ud ~ uniform
        ud = torch.randint(10, size=(n, 1))
        # d <- ud,
        d = ud.int()
        # c = (ud >= 5) \xor uc, p(uc=1) = 0.75
        c = torch.logical_xor(ud >= 5, torch.bernoulli(0.75 * torch.ones((n, 1)))).int()
        # bc = c \xor uc, p(ubc=1) = 0.4
        bc = torch.logical_xor(c, torch.bernoulli(0.4 * torch.ones((n, 1)))).int()
        # t_1 = (d >= 5) \and u_4, p(u4=1) = 0.6
        t_1 = torch.logical_and(d >= 5, torch.bernoulli(0.75 * torch.ones((n, 1)))).int()
        # t_1 = (d < 5) \and u_5, p(u5=1) = 0.2
        t_2 = torch.logical_and(d < 5, torch.bernoulli(0.1 * torch.ones((n, 1)))).int()
        bw = torch.logical_xor(torch.logical_xor(t_1, t_2).int(), c)
        v = dict()
        v['Digit'] = d
        v['DigitColor'] = c
        v['BarColor'] = bc
        v['BarWidth'] = bw
        return v
    else:
        raise RuntimeError('The graph is not implemented.')


def color_barmnist_fI(d, c, bc, bw, raw_mnist, threshold=180):
    total = len(raw_mnist[d.item()])
    ind = torch.randint(total, size=[1])
    I = raw_mnist[d.item()][ind].clone().numpy().reshape(28, 28, 1)
    dtype = I.dtype
    if c == 0:  # c = 0: red
        arr = np.concatenate([I,
                              np.zeros((28, 28, 1), dtype=dtype),
                              np.zeros((28, 28, 1), dtype=dtype)], axis=2)
    else:  # c = 0: green
        arr = np.concatenate([np.zeros((28, 28, 1), dtype=dtype),
                              I,
                              np.zeros((28, 28, 1), dtype=dtype)], axis=2)

    if bw == 0 and bc == 0:
        arr[0:4, :, 0] = np.where(arr[0:4, :, 0] < threshold, threshold, arr[0:4, :, 0])
    elif bw == 0 and bc == 1:
        arr[0:4, :, 1] = np.where(arr[0:4, :, 1] < threshold, threshold, arr[0:4, :, 1])
    elif bw == 1 and bc == 0:
        arr[0:10, :, 0] = np.where(arr[0:10, :, 0] < threshold, threshold, arr[0:10, :, 0])
    elif bw == 1 and bc == 1:
        arr[0:10, :, 1] = np.where(arr[0:10, :, 1] < threshold, threshold, arr[0:10, :, 1])

    return arr


class ColoredBarMNIST(datasets.VisionDataset):
    def __init__(self, cg, root, env='train', transform=None, target_transform=None, ow=False, v_only=False):
        super(ColoredBarMNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)

        self.v_only = v_only
        self.prepare_bar_mnist(ow, cg, env)
        if env in ['train', 'test']:
            self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredBarMNIST', cg, env) + '.pt')
        else:
            raise RuntimeError(f'{env} unknown. Valid envs are train, test')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, generative factors) where target is index of the generative factors [digit, bar, color]
        """
        img, target = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.v_only:
            return target

        return img, target

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_bar_mnist(self, ow, cg, env):
        if not os.path.isdir(os.path.join(self.root, 'ColoredBarMNIST')):
            os.mkdir(os.path.join(self.root, 'ColoredBarMNIST'))
        bar_mnist_dir = os.path.join(self.root, 'ColoredBarMNIST', cg)
        if os.path.exists(os.path.join(bar_mnist_dir, env+'.pt')) and not ow:

            print('Colored Bar MNIST' + cg + ' dataset already exists')
            return

        print('Preparing Colored Bar MNIST')
        print('Causal Diagram:' + cg)

        if env not in ['train', 'test']:
            raise RuntimeError(f'{env} unknown. Valid envs are train and test')

        if env == 'train':
            mnist_data = datasets.mnist.MNIST(self.root, train=True, download=True)
        else:
            mnist_data = datasets.mnist.MNIST(self.root, train=False, download=True)

        images = mnist_data.data
        labels = mnist_data.targets

        raw_mnist_n = len(images)
        raw_mnist_images = dict()
        for i in range(len(labels)):
            if labels[i].item() not in raw_mnist_images:
                raw_mnist_images[labels[i].item()] = []
            raw_mnist_images[labels[i].item()].append(images[i])

        v_sample = gen_scm(cg, raw_mnist_n)

        dat_set = []
        for i in range(raw_mnist_n):
            arr = color_barmnist_fI(v_sample['Digit'][i], v_sample['DigitColor'][i], v_sample['BarColor'][i],
                                    v_sample['BarWidth'][i], raw_mnist_images)
            label_oh = torch.zeros(10)
            label_oh[int(v_sample['Digit'][i])] = 1
            dc_oh = torch.zeros(2)
            dc_oh[int(v_sample['DigitColor'][i])] = 1
            bc_oh = torch.zeros(2)
            bc_oh[int(v_sample['BarColor'][i])] = 1
            bw_oh = torch.zeros(2)
            bw_oh[int(v_sample['BarWidth'][i])] = 1
            dat_set.append((Image.fromarray(arr),
                            torch.cat((label_oh.view(-1), dc_oh.view(-1), bc_oh.view(-1), bw_oh.view(-1)))))

        if not os.path.isdir(bar_mnist_dir):
            os.mkdir(bar_mnist_dir)
        torch.save(dat_set, os.path.join(bar_mnist_dir, env+'.pt'))


if __name__ == "__main__":
    cg = 'full-ncm'
    ColoredBarMNIST(cg=cg, root='../../dat/img', ow=False)
    trans_f = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                  ])
    batch_size = 1000
    train_set = ColoredBarMNIST(cg=cg, root='../../dat/img', env='train', transform=trans_f)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               drop_last=True)

    y_tol = []
    for idx, (x, y) in enumerate(train_loader):
        if idx == 0:
            save_image(
                x[:20, :, :, :],
                cg + "_samples.png",
                pad_value=2
            )
            print(y[:20, :])
        y_tol.append(y)


    y_tol = torch.cat(y_tol, dim=0)
    dat = {
        'D': torch.argmax(y_tol[:, :10], dim=1, keepdim=True),
        'C': torch.argmax(y_tol[:, 10:12], dim=1, keepdim=True),
        'BC': torch.argmax(y_tol[:, 12:14], dim=1, keepdim=True),
        'BW': torch.argmax(y_tol[:, 14:16], dim=1, keepdim=True),
    }

    cols = dict()
    for v in sorted(dat):
        result = dat[v].detach().cpu().numpy()
        for i in range(result.shape[1]):
            cols["{}{}".format(v, i)] = np.squeeze(result[:, i])

    df = pd.DataFrame(cols)
    grouped = (df.groupby(list(df.columns))
                .apply(lambda x: len(x) / len(df))
                .rename('P(V)').reset_index()
                [[*df.columns, 'P(V)']])
    ordered_cols = ['D0', 'C0', 'BC0', 'BW0', 'P(V)']
    grouped = grouped[ordered_cols]
    grouped = grouped.sort_values(['D0', 'C0', 'BC0', 'BW0']).reset_index(drop=True)
    print(grouped[grouped['D0'] == 9])

    # idx = []
    # n = len(y_tol)
    # y_tol = torch.cat(y_tol, dim=0)
    # for i in range(y_tol.size(0)):
    #     if y_tol[i, 0] == 1 and y_tol[i, 10] == 1 and y_tol[i, 12] == 1:
    #         idx.append(i)
    # print(len(idx))
    # print(torch.sum(y_tol[idx, 14]) / len(idx))
