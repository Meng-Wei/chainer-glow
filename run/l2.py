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
from optimizer import Optimizer

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

    if False:
        assert args.dataset_format in ["png", "npy"]

        files = Path(args.dataset_path).glob("*.{}".format(args.dataset_format))
        if args.dataset_format == "png":
            images = []
            for filepath in files:
                image = np.array(Image.open(filepath)).astype("float32")
                image = preprocess(image, hyperparams.num_bits_x)
                images.append(image)
            assert len(images) > 0
            images = np.asanyarray(images)
        elif args.dataset_format == "npy":
            images = []
            for filepath in files:
                array = np.load(filepath).astype("float32")
                array = preprocess(array, hyperparams.num_bits_x)
                images.append(array)
            assert len(images) > 0
            num_files = len(images)
            images = np.asanyarray(images)
            images = images.reshape((num_files * images.shape[1], ) +
                                    images.shape[2:])
        else:
            raise NotImplementedError

        dataset = glow.dataset.Dataset(images)
        iterator = glow.dataset.Iterator(dataset, batch_size=1)

        print(tabulate([["#image", len(dataset)]]))

    # TODO: Init encoder and stored info
    encoder = Glow(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        encoder.to_gpu()

    # Load picture
    x = np.array(Image.open('bg/1.png')).astype('float32')
    x = preprocess(x, hyperparams.num_bits_x)
    # img_x = make_uint8(x, num_bins_x)
    # img_x = Image.fromarray(img_x)
    # img_x.save('x.png')

    x = to_gpu(xp.expand_dims(x, axis=0))
    x += xp.random.uniform(0, 1.0/num_bins_x, size=x.shape)

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
    class eps(chainer.ChainList):
        def __init__(self, shape, glow_encoder):
            super().__init__()
            self.encoder = glow_encoder

            with self.init_scope():
                self.b = chainer.Parameter(initializers.Normal(), shape)
                # self.m = chainer.Parameter(initializers.Uniform(), (shape[2], shape[3]))
                self.m = chainer.Parameter(initializers.Uniform(), (16, 16))
        
        def forward(self, x):
            b = cf.tanh(self.b)
            m = cf.repeat(self.m, 8, axis=1)
            m = cf.repeat(m, 8, axis=0)
            # for i in range(self.m.shape[0]):
            #     for j in range(self.m.shape[1]):
            #         b[:, i*8:(i+1)*8, j*8:(j+1)*8] *= self.m[i, j]

            b = b * m
            cur_x = cf.add(x, b)

            z, logdet = self.encoder.forward_step(cur_x)

            ez = []
            for (zi, mean, ln_var) in z:
                ez.append(zi.data.reshape(-1,))
            ez = np.concatenate(ez)

            # return ez, z, logdet, cf.batch_l2_norm_squared(self.b), self.b * 1, cur_x, self.m*1
            return ez, z, logdet, cf.batch_l2_norm_squared(b), b, cur_x, m

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

    optimizer = Optimizer(epsilon)
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
        z, zs, fw_ldt, b_norm, b, cur_x, m = epsilon.forward(x)   
        print('b_norm, ', b_norm.data)
        print('test ', xp.linalog.norm(b.data*m.data)**2 )

        fw_ldt -= math.log(num_bins_x) * num_pixels

        logpZ1 = 0
        for (zi, mean, ln_var) in zs:
            logpZ1 += cf.gaussian_nll(zi, mean, ln_var)
            
        logpZ2 = cf.gaussian_nll(z, xp.zeros(z.shape), xp.zeros(z.shape)).data
        # logpZ2 = cf.gaussian_nll(z, np.mean(z), np.log(np.var(z))).data

        logpZ = (4*logpZ2 + logpZ1)/5
        # loss =  1000* b_norm + logpZ * 0.5 - fw_ldt
        loss = b_norm + 0.01 * (logpZ - fw_ldt)

        # print(b_norm, xp.linalg.norm(b.data))
        epsilon.cleargrads()
        loss.backward()
        optimizer.update(training_step)
        training_step += 1

        z_s.append(z.get())
        b_s.append(cupy.asnumpy(b.data))
        m_s.append(cupy.asnumpy(m.data))
        loss_s.append(_float(loss))
        logpZ_s.append(_float(logpZ))
        logDet_s.append(_float(fw_ldt))

        printr(
            "Iteration {}: loss: {:.6f} - logpZ: {:.6f} - log_det: {:.6f} - logpX: {:.6f}\n".
            format(
                iteration + 1,
                _float(loss),
                _float(logpZ),
                _float(fw_ldt),
                _float(logpZ) - _float(fw_ldt)
            )
        )

        if iteration % 100 == 9:
            np.save(args.ckpt + '/'+str(j)+'z.npy', z_s)
            np.save(args.ckpt + '/'+str(j)+'b.npy', b_s)
            np.save(args.ckpt + '/'+str(j)+'loss.npy', loss_s)
            np.save(args.ckpt + '/'+str(j)+'logpZ.npy', logpZ_s)
            np.save(args.ckpt + '/'+str(j)+'logDet.npy', logDet_s)
            cur_x = make_uint8(cur_x[0].data, num_bins_x)
            np.save(args.ckpt + '/'+str(j)+'image.npy', cur_x)
            np.save(args.ckpt + '/'+str(j)+'m.npy', m_s)
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
    parser.add_argument("--gpu-device", "-gpu", type=int, default=1)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument("--total-iteration", "-iter", type=int, default=10)
    args = parser.parse_args()
    main()
