import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import NetsBlock as NetsBlock
import torch as th
from math import pi
from math import log2
import time
import math


class INRNetsGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='instanceaffine')
        parser.set_defaults(lr_instance=True)
        parser.set_defaults(no_instance_dist=True)
        parser.set_defaults(hr_coor="cosine")
        return parser

    def __init__(self, hr_stream=None, lr_stream=None, fast=False):
        super(INRNetsGenerator, self).__init__()
        self.contain_dontcare_label = False
        self.no_instance_edge = True
        self.no_instance_dist = True
        self.label_nc = 13
        self.lr_instance = False
        self.learned_ds_factor = 16
        self.gpu_ids = [0]
        if lr_stream is None or hr_stream is None:
            lr_stream = dict()
            hr_stream = dict()
        self.num_inputs = self.label_nc + (1 if self.contain_dontcare_label else 0) + (0 if (self.no_instance_edge & self.no_instance_dist) else 1)
        self.lr_instance = self.lr_instance
        self.learned_ds_factor = self.learned_ds_factor  # (S2 in sec. 3.2)
        self.gpu_ids = self.gpu_ids

        # calculates the total downsampling factor in order to get the final grid of parameters
        self.crop_size = 256
        self.ratio = 1.0
        self.downsampling = self.crop_size // (16 * self.ratio)
        self.downsampling = 4
        self.grid_stream = GRIDStream(self.downsampling, num_inputs=3,
                                      num_outputs=31, width=64,
                                      depth=5, coordinates="cosine",
                                      no_one_hot=False, lr_instance=False,
                                      **hr_stream)

        num_params = self.grid_stream.num_params
        num_inputs_lr = self.grid_stream.num_inputs + (1 if self.lr_instance else 0)
        norm_layer = get_nonspade_norm_layer("opt", "instanceaffine")
        self.hyper = HyperNet(3, num_params, norm_layer, width=64,
                              max_width=1024, depth=5,
                              learned_ds_factor=4,
                              reflection_pad=False, **lr_stream)

    def use_gpu(self):
        return len(self.gpu_ids) > 0

    def forward(self, inp):
        hy_features = self.hyper(inp)
        output = self.grid_stream(inp, hy_features)
        return output


def _get_coords(bs, h, w, device, coords_type):
    """Creates the position encoding for the MLPs"""
    if coords_type == 'cosine':
        f0 = 4
        f = f0
        while f > 1:
            x = th.arange(0, w).float() / w
            y = th.arange(0, h).float() / h
            n = log2(f)
            xcos = th.cos((2 * pi * th.remainder(x, f).float() / f).float())
            xsin = th.sin((2 * pi * th.remainder(x, f).float() / f).float())
            ycos = th.cos((2 * pi * th.remainder(y, f).float() / f).float())
            ysin = th.sin((2 * pi * th.remainder(y, f).float() / f).float())
            # xcos = th.cos((2 * pi * n * x).float())
            # xsin = th.sin((2 * pi * n * x).float())
            # ycos = th.cos((2 * pi * n * y).float())
            # ysin = th.sin((2 * pi * n * y).float())
            xcos = xcos.view(1, 1, 1, w).repeat(bs, 1, h, 1)
            xsin = xsin.view(1, 1, 1, w).repeat(bs, 1, h, 1)
            ycos = ycos.view(1, 1, h, 1).repeat(bs, 1, 1, w)
            ysin = ysin.view(1, 1, h, 1).repeat(bs, 1, 1, w)
            coords_cur = th.cat([xcos, xsin, ycos, ysin], 1).to(device)
            if f < f0:
                coords = th.cat([coords, coords_cur], 1).to(device)
            else:
                coords = coords_cur
            f = f//2
    else:
        raise NotImplementedError()
    return coords.to(device)


class HyperNet(th.nn.Sequential):
    """Convolutional LR stream to estimate the pixel-wise MLPs parameters"""
    def __init__(self, num_in, num_out, norm_layer, width=64, max_width=1024, depth=7, learned_ds_factor=16,
                 reflection_pad=False, replicate_pad=False):
        super(HyperNet, self).__init__()

        model = []

        self.num_out = num_out
        padw = 1
        if reflection_pad:
            padw = 0
            model += [th.nn.ReflectionPad2d(1)]
        if replicate_pad:
            padw = 0
            model += [th.nn.ReplicationPad2d(1)]

        count_ly = 0

        model += [norm_layer(th.nn.Conv2d(num_in, width, 3, stride=1, padding=padw)),
                  th.nn.ReLU(inplace=True)]

        num_ds_layers = int(log2(learned_ds_factor))

        for i in range(num_ds_layers):
            if reflection_pad:
                model += [th.nn.ReflectionPad2d(1)]
            if replicate_pad:
                model += [th.nn.ReplicationPad2d(1)]
            if i == num_ds_layers-1:
                model += [norm_layer(th.nn.Conv2d(width, width, 3, stride=1, padding=padw)),
                      th.nn.ReLU(inplace=True)]
                model += [norm_layer(th.nn.Conv2d(width, width, 3, stride=1, padding=padw)),
                      th.nn.ReLU(inplace=True)]

                last_width = max_width
                model += [norm_layer(th.nn.Conv2d(width, last_width, 3, stride=2, padding=padw)),
                          th.nn.ReLU(inplace=True)]
                width = last_width
            else:
                model += [norm_layer(th.nn.Conv2d(width, width, 3, stride=2, padding=padw)),
                      th.nn.ReLU(inplace=True)]

        # ConvNet to estimate the MLPs parameters"
        for i in range(count_ly, count_ly+depth):
            model += [NetsBlock(width, norm_layer, reflection_pad=reflection_pad, replicate_pad=replicate_pad)]

        # Final parameter prediction layer, transfer conv channels into the per-pixel number of MLP parameters
        model += [th.nn.Conv2d(width, self.num_out, 1)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


class GRIDStream(th.nn.Module):
    """Addaptive pixel-wise MLPs"""
    def __init__(self, downsampling,
                 num_inputs=13, num_outputs=3, width=64, depth=5, coordinates="cosine",
                 no_one_hot=False, lr_instance=False):
        super(GRIDStream, self).__init__()

        self.lr_instance = lr_instance
        self.downsampling = downsampling
        self.num_inputs = num_inputs - (1 if self.lr_instance else 0)
        self.num_outputs = num_outputs
        self.width = width
        self.depth = depth
        self.coordinates = coordinates
        self.xy_coords = None
        self.no_one_hot = no_one_hot
        self.channels = []
        self._set_channels()

        self.num_params = 0
        self.splits = {}
        self._set_num_params()

    @property  # for backward compatibility
    def ds(self):
        return self.downsampling

    def _set_channels(self):
        """Compute and store the hr-stream layer dimensions."""
        in_ch = self.num_inputs
        if self.coordinates == "cosine":
            in_ch += int(4*log2(self.downsampling))
            # L = 8
            # in_ch += int(L*4 + 2)
        self.channels = [in_ch]
        for _ in range(self.depth - 1):  # intermediate layer -> cste size
            self.channels.append(self.width)
        # output layer
        self.channels.append(self.num_outputs)

    def _set_num_params(self):
        nparams = 0
        self.splits = {
            "biases": [],
            "weights": [],
        }

        # go over input/output channels for each layer
        idx = 0
        for layer, nci in enumerate(self.channels[:-1]):
            nco = self.channels[layer + 1]
            nparams += nco  # FC biases
            self.splits["biases"].append((idx, idx + nco))
            idx += nco

            nparams += nci * nco  # FC weights
            self.splits["weights"].append((idx, idx + nco * nci))
            idx += nco * nci

        self.num_params = nparams

    def _get_weight_indices(self, idx):
        return self.splits["weights"][idx]

    def _get_bias_indices(self, idx):
        return self.splits["biases"][idx]

    def forward(self, inp, hy_features):
        assert hy_features.shape[1] == self.num_params, "incorrect input params"
        if self.lr_instance:
            inp = inp[:, :-1, :, :]

        # Fetch sizes
        k = int(self.downsampling)
        k = 4
        bs, _, h, w = inp.shape
        bs, _, h_lr, w_lr = hy_features.shape

        # Spatial encoding
        if not(self.coordinates is None):
            if self.xy_coords is None:
                self.xy_coords = _get_coords(bs, h, w, inp.device, self.coordinates)
            inp = th.cat([inp, self.xy_coords], 1)

        # all pixels within a tile of kxk are processed by the same MLPs parameters
        nci = inp.shape[1]
        # bs, 5 rgbxy, h//k=h_lr, w//k=w_lr, k, k
        tiles = inp.unfold(2, k, k).unfold(3, k, k)
        tiles = tiles.permute(0, 2, 3, 4, 5, 1).contiguous().view(
            bs, h_lr, w_lr, int(k * k), nci)
        out = tiles
        num_layers = len(self.channels) - 1

        for idx, nci in enumerate(self.channels[:-1]):
            nco = self.channels[idx + 1]

            # Select params in lowres buffer
            bstart, bstop = self._get_bias_indices(idx)
            wstart, wstop = self._get_weight_indices(idx)

            w_ = hy_features[:, wstart:wstop]
            b_ = hy_features[:, bstart:bstop]

            w_ = w_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, nci, nco)
            b_ = b_.permute(0, 2, 3, 1).view(bs, h_lr, w_lr, 1, nco)
            out = th.matmul(out, w_) + b_

            # Apply RelU non-linearity in all but the last layer, and tanh in the last
            if idx < num_layers - 1:
                out = th.nn.functional.leaky_relu(out, 0.1, inplace=True)
            else:
                # out = torch.nn.Sigmoid()(out)
                out = out

        # reorder the tiles in their correct position, and put channels first
        out = out.view(bs, h_lr, w_lr, k, k, self.num_outputs).permute(
            0, 5, 1, 3, 2, 4)
        out = out.contiguous().view(bs, self.num_outputs, h, w)
        # out = torch.nn.Tanh()(out)

        return out


def TV_Loss(x):
    TVLoss_weight = 0.1
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = x[:, :, 1:, :].size()[1] * x[:, :, 1:, :].size()[2] * x[:, :, 1:, :].size()[3]
    count_w = x[:, :, :, 1:].size()[1] * x[:, :, :, 1:].size()[2] * x[:, :, :, 1:].size()[3]
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
    return TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size