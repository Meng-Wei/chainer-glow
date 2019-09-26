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

    images_list = []
    enc_z_list = []
    log_det_list = []
    logpZ_list = []
    logpZ2_list = []


    # Load picture
    x = to_gpu(np.array(Image.open('bg/1.png')))
    x = preprocess(x, num_bins_x)
    x = xp.expand_dims(x, axis=0)
    ori_x = x + xp.random.uniform(0, 1.0 / num_bins_x, size=x.shape)
    ori_x = xp.array(ori_x, dtype='float32')

    # Construct epsilon
    class eps(chainer.ChainList):
        def __init__(self, shape, glow_encoder):
            super().__init__()
            self.encoder = glow_encoder

            with self.init_scope():
                self.b = chainer.Parameter(
                    initializers.Normal(), shape)
                # self.b = chainer.Parameter(
                #     initializers.Normal(dtype='float64'), shape)
        
        def forward(self, x):
            cur_x = cf.add(x, self.b)
            z, log_det = self.encoder.forward_step(cur_x)
            return z, log_det, cf.batch_l2_norm_squared(self.b), self.b * 1
        def save(self, path):
            filename = 'l1_model.hdf5'
            self.save_parameter(path, filename, self)
        def save_parameter(self, path, filename, params):
            tmp_filename = str(uuid.uuid4())
            tmp_filepath = os.path.join(path, tmp_filename)
            save_hdf5(tmp_filepath, params)
            os.rename(tmp_filepath, os.path.join(path, filename))


    epsilon = eps(ori_x.shape, encoder)
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
    j = 0

    for iteration in range(args.total_iteration):
        start_time = time.time()

        # ori_x += epsilon
        # z, fw_ldt = encoder.forward_step(ori_x)
        z, fw_ldt, b_l2norm, b = epsilon.forward(ori_x)

        # logpZ = 0
        ez = []
        for (zi, mean, ln_var) in z:
            # logpZ += cf.gaussian_nll(zi, mean, ln_var)
            ez.append(zi.data.reshape(-1,))
            
        ez = np.concatenate(ez)
        logpZ = cf.gaussian_nll(ez, xp.zeros(ez.shape), xp.zeros(ez.shape)).data
        # logpZ = cf.gaussian_nll(ez, np.mean(ez), np.log(np.var(ez))).data

        loss = b_l2norm[0] + (logpZ - fw_ldt)

        # print("loss", _float(loss), loss.data)
        # print('logpZ', _float(logpZ), logpZ.data)
        # print('logDet', _float(fw_ldt), fw_ldt)

        epsilon.cleargrads()
        loss.backward()
        optimizer.update(training_step)
        training_step += 1

        z_s.append(ez.get())
        b_s.append(cupy.asnumpy(b.data))
        loss_s.append(_float(loss))
        logpZ_s.append(_float(logpZ))
        logDet_s.append(_float(fw_ldt))

        printr(
            "Iteration {}: Batch {} - loss: {:.8f} - logpZ: {:.8f} - log_det: {:.8f}\n".
            format(
                iteration + 1, 1,
                _float(loss),
                _float(logpZ),
                _float(fw_ldt)
            )
        )

        

        if iteration % 100 == 99:
            np.save('logs/'+str(j)+'z.npy', z_s)
            np.save('logs/'+str(j)+'b.npy', b_s)
            np.save('logs/'+str(j)+'loss.npy', loss_s)
            np.save('logs/'+str(j)+'logpZ.npy', logpZ_s)
            np.save('logs/'+str(j)+'logDet.npy', logDet_s)
            z_s = []
            b_s = []
            loss_s = []
            logpZ_s = []
            logDet_s = []
            j += 1
            epsilon.save(args.ckpt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, default='/home/data1/meng/chainer/snapshot_64')
    parser.add_argument("--gpu-device", "-gpu", type=int, default=1)
    parser.add_argument('--ckpt', type=str, default='logs')
    # parser.add_argument("--dataset-path", "-dataset", type=str, required=False)
    # parser.add_argument("--dataset-format", "-ext", type=str, required=True)
    parser.add_argument("--total-iteration", "-iter", type=int, default=10)
    args = parser.parse_args()
    main()
