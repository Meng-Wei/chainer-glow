import os
import sys
import argparse
import math
import random
import time

from tabulate import tabulate
from PIL import Image
from pathlib import Path

import chainer
import chainer.functions as cf
import cupy
import numpy as np
from chainer.backends import cuda
from chainer import initializers
from chainer.serializers import load_hdf5, save_hdf5
import uuid

sys.path.append(".")
sys.path.append("..")
import glow

from model import Glow, to_cpu, to_gpu
from hyperparams import Hyperparameters
# from optimizer import Optimizer
from chainer import optimizers

def make_uint8(array, bins):
    if array.ndim == 4:
        array = array[0]
    if (array.shape[2] == 3):
        return np.uint8(
            np.clip(
                np.floor((to_cpu(array) + 0.5) * bins) * (255 / bins), 0, 255))
    return np.uint8(
        np.clip(
            np.floor((to_cpu(array.transpose(1, 2, 0)) + 0.5) * bins) *
            (255 / bins), 0, 255))

def preprocess(image, num_bits_x):
    num_bins_x = 2**num_bits_x
    if num_bits_x < 8:
        image = np.floor(image / (2**(8 - num_bits_x)))
    image = image / num_bins_x - 0.5
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))
    elif image.ndim == 4:
        image = image.transpose((0, 3, 1, 2))
    else:
        raise NotImplementedError
    return image

def merge_factorized_z(factorized_z, factor=2):
    z = None
    for zi in reversed(factorized_z):
        xp = cuda.get_array_module(zi.data)
        z = zi.data if z is None else xp.concatenate((zi.data, z), axis=1)
        z = glow.nn.functions.unsqueeze(z, factor, xp)
    return z

def printr(string):
    sys.stdout.write(string)
    sys.stdout.write("\r")

def _float(v):
    if isinstance(v, float):
        return v
    if isinstance(v, chainer.Variable):
        return float(v.data)
    return float(v)

def main():
    try:
        os.mkdir(args.ckpt)
    except:
        pass

    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    hyperparams = Hyperparameters(args.snapshot_path)
    hyperparams.print()

    num_bins_x = 2.0**hyperparams.num_bits_x
    num_pixels = 3 * hyperparams.image_size[0] * hyperparams.image_size[1]

    encoder = Glow(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        encoder.to_gpu()

    # Load picture
    x = np.array(Image.open(args.img) ).astype('float32')
    x = preprocess(x, hyperparams.num_bits_x)

    x = to_gpu(xp.expand_dims(x, axis=0))
    x += xp.random.uniform(0, 1.0/num_bins_x, size=x.shape)

    # # Print this image info:
    # z, fw_ldt = encoder.forward_step(x)        
    # fw_ldt -= math.log(num_bins_x) * num_pixels
    
    # logpZ = 0
    # ez = []
    # for (zi, mean, ln_var) in z:
    #     logpZ += cf.gaussian_nll(zi, mean, ln_var)
    #     ez.append(zi.data.reshape(-1,))
    # ez = np.concatenate(ez)
    # logpZ2 = cf.gaussian_nll(ez, xp.zeros(ez.shape), xp.zeros(ez.shape)).data

    # print(fw_ldt, logpZ, logpZ2)

    # Construct epsilon
    class eps(chainer.Chain):
        def __init__(self, shape, glow_encoder):
            super().__init__()
            self.encoder = glow_encoder

            with self.init_scope():
                self.b = chainer.Parameter(initializers.Zero(), shape)
                self.m = chainer.Parameter(initializers.One(), (3, 8, 8))
        
        def forward(self, x):
            b_ = cf.tanh(self.b)

            # Not sure if implementation is wrong
            m_ = cf.softplus(self.m)
            # m = cf.repeat(m, 8, axis=2)
            # m = cf.repeat(m, 8, axis=1)
            m_ = cf.repeat(m_, 16, axis=2)
            m_ = cf.repeat(m_, 16, axis=1)

            b_ = b_ * m_ 
            x_ = cf.add(x, b_)
            x_ = cf.clip(x_, -0.5, 0.5)

            z = []
            zs, logdet = self.encoder.forward_step(x_)
            for (zi, mean, ln_var) in zs:
                z.append(zi)

            z = merge_factorized_z(z)

            return z, zs, logdet, xp.sum(xp.abs(b_.data)), xp.tanh(self.b.data * 1), m_, x_

        def save(self, path):
            filename = 'loss_model.hdf5'
            self.save_parameter(path, filename, self)

        def save_parameter(self, path, filename, params):
            tmp_filename = str(uuid.uuid4())
            tmp_filepath = os.path.join(path, tmp_filename)
            save_hdf5(tmp_filepath, params)
            os.rename(tmp_filepath, os.path.join(path, filename))

    epsilon = eps(x.shape, encoder)
    if using_gpu:
        epsilon.to_gpu()

    # optimizer = Optimizer(epsilon)
    optimizer = optimizers.Adam(alpha=0.0005).setup(epsilon)
    # optimizer = optimizers.SGD().setup(epsilon)
    epsilon.b.update_rule.hyperparam.lr = 0.0001
    epsilon.m.update_rule.hyperparam.lr = 0.1
    print('init finish')

    training_step = 0

    z_s = []
    b_s = []
    loss_s = []
    logpZ_s = []
    logDet_s = []
    m_s = []
    j = 0

    for iteration in range(args.total_iteration):
        z, zs, fw_ldt, b_norm, cur_b, cur_m, cur_x = epsilon.forward(x)

        epsilon.cleargrads()

        # Construct loss term:
        fw_ldt -= math.log(num_bins_x) * num_pixels

        logpZ1 = 0
        factor_z = []
        for (zi, mean, ln_var) in zs:
            factor_z.append(zi.data)
            logpZ1 += cf.gaussian_nll(zi, mean, ln_var)
            
        logpZ2 = cf.gaussian_nll(z, xp.zeros(z.shape), xp.zeros(z.shape)).data
        # logpZ2 = cf.gaussian_nll(z, np.mean(z), np.log(np.var(z))).data

        logpZ = (logpZ2 + logpZ1) * 0.5
        loss = 10 * b_norm + (logpZ - fw_ldt)

        loss.backward()
        optimizer.update()
        training_step += 1

        z_s.append(z.get())
        b_s.append(cupy.asnumpy(cur_b))
        m_s.append(cupy.asnumpy(cur_m.data))
        loss_s.append(_float(loss))
        logpZ_s.append(_float(logpZ))
        logDet_s.append(_float(fw_ldt))

        printr(
            "Iteration {}: loss: {:.6f} - b_norm: {:.6f} - logpZ: {:.6f} - logpZ1: {:.6f} - logpZ2: {:.6f} - log_det: {:.6f} - logpX: {:.6f}\n".
            format(
                iteration + 1,
                _float(loss),
                _float(b_norm),
                _float(logpZ),
                _float(logpZ1),
                _float(logpZ2),
                _float(fw_ldt),
                _float(logpZ) - _float(fw_ldt)
            )
        )

        if iteration % 100 == 99:
            np.save(args.ckpt + '/'+str(j)+'z.npy', z_s)
            np.save(args.ckpt + '/'+str(j)+'b.npy', b_s)
            np.save(args.ckpt + '/'+str(j)+'loss.npy', loss_s)
            np.save(args.ckpt + '/'+str(j)+'logpZ.npy', logpZ_s)
            np.save(args.ckpt + '/'+str(j)+'logDet.npy', logDet_s)
            cur_x = make_uint8(cur_x[0].data, num_bins_x)
            np.save(args.ckpt + '/'+str(j)+'image.npy', cur_x)
            np.save(args.ckpt + '/'+str(j)+'m.npy', m_s)
            
            with encoder.reverse() as decoder:
                rx, _ = decoder.reverse_step(factor_z)
                rx_img = make_uint8(rx.data[0], num_bins_x)
                np.save(args.ckpt + '/'+str(j)+'res.npy', rx_img)

            z_s = []
            b_s = []
            loss_s = []
            logpZ_s = []
            logDet_s = []
            m_s = []
            j += 1
            epsilon.save(args.ckpt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, default='/home/data1/meng/chainer/snapshot_128')
        # "--snapshot-path", "-snapshot", type=str, default='/home/data1/meng/chainer/snapshot_64')
    parser.add_argument("--gpu-device", "-gpu", type=int, default=1)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument("--total-iteration", "-iter", type=int, default=1000)
    parser.add_argument("-img", type=str, required=True)
    args = parser.parse_args()
    main()
