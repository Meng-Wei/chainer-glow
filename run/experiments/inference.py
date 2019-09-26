import os
import sys
import argparse
import math

import chainer
import chainer.functions as cf
import cupy
import numpy as np
import matplotlib.pyplot as plt

from chainer.backends import cuda
from tabulate import tabulate
from PIL import Image
from pathlib import Path

sys.path.append(os.path.join("..", ".."))
import glow

sys.path.append("..")
from model import Glow, to_cpu, to_gpu
from hyperparams import Hyperparameters


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


def main():
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    hyperparams = Hyperparameters(args.snapshot_path)
    hyperparams.print()

    num_bins_x = 2.0**hyperparams.num_bits_x
    num_pixels = 3 * hyperparams.image_size[0] * hyperparams.image_size[1]
    
    # Get Dataset:
    if True:
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

    encoder = Glow(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        encoder.to_gpu()

    ori_x = []
    enc_z = []
    rev_x = []
    fw_logdet = []
    logpZ = []
    logpZ2 = []
    i = 0

    with chainer.no_backprop_mode() and encoder.reverse() as decoder:
        for data_indices in iterator:
            i += 1

            x = to_gpu(dataset[data_indices]) # 1x3x64x64
            x += xp.random.uniform(0, 1.0 / num_bins_x, size=x.shape)
            x_img = make_uint8(x[0], num_bins_x)
            ori_x.append(x_img)

            factorized_z_distribution, fw_ldt = encoder.forward_step(x)
            fw_ldt -= math.log(num_bins_x) * num_pixels
            fw_logdet.append(cupy.asnumpy(fw_ldt.data))

            factor_z = []
            ez = []
            nll = 0
            for (zi, mean, ln_var) in factorized_z_distribution:
                nll += cf.gaussian_nll(zi, mean, ln_var)
                factor_z.append(zi.data)
                ez.append(zi.data.reshape(-1,))
            
            ez = np.concatenate(ez)
            enc_z.append(ez.get())
            logpZ.append(cupy.asnumpy(nll.data))
            logpZ2.append(
                cupy.asnumpy(cf.gaussian_nll(ez, np.mean(ez), np.log(np.var(ez))).data )
            )

            rx, _ = decoder.reverse_step(factor_z)
            rx_img = make_uint8(rx.data[0], num_bins_x)
            rev_x.append(rx_img)

            if i % 100 == 0:
                np.save(str(i)+'/ori_x.npy', ori_x)
                fw_logdet = np.array(fw_logdet)
                np.save(str(i)+'/fw_logdet.npy', fw_logdet)
                np.save(str(i)+'/enc_z.npy', enc_z)
                logpZ = np.array(logpZ)
                np.save(str(i)+'/logpZ.npy', logpZ)
                logpZ2 = np.array(logpZ2)
                np.save(str(i)+'/logpZ2.npy', logpZ2)
                np.save(str(i)+'/rev_x.npy', rev_x)

                ori_x = []
                enc_z = []
                rev_x = []
                fw_logdet = []
                logpZ = []
                logpZ2 = []
                return
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--dataset-path", "-dataset", type=str, required=True)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=1)
    parser.add_argument("--dataset-format", "-ext", type=str, required=True)
    args = parser.parse_args()
    main()
