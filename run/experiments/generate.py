import os
import sys
import argparse

import chainer
import chainer.functions as cf
import cupy
import numpy as np
import matplotlib.pyplot as plt

from chainer.backends import cuda
from tabulate import tabulate

sys.path.append(os.path.join("..", ".."))
import glow

sys.path.append("..")
from model import Glow, to_cpu
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


def main():
    xp = np
    using_gpu = args.gpu_device >= 0
    if using_gpu:
        cuda.get_device(args.gpu_device).use()
        xp = cupy

    hyperparams = Hyperparameters(args.snapshot_path)
    hyperparams.print()

    num_bins_x = 2.0**hyperparams.num_bits_x

    encoder = Glow(hyperparams, hdf5_path=args.snapshot_path)
    if using_gpu:
        encoder.to_gpu()

    with chainer.no_backprop_mode() and encoder.reverse() as decoder:
        # while True:
        #     z = xp.random.normal(
        #         0, args.temperature, size=(
        #             1,
        #             3,
        #         ) + hyperparams.image_size).astype("float32")

        #     x, _ = decoder.reverse_step(z)
        #     x_img = make_uint8(x.data[0], num_bins_x)
        #     plt.imshow(x_img, interpolation="none")
        #     plt.pause(.01)

        i = 0
        j = 0

        enc_z = []
        rev_x = []
        bk_logdet = []
        logpZ2 = []
        fw_logdet = []
        sec_z = []
        sec_pz = []
        sec_pz2 = []

        while j < 4: 
            j += 1

            z = xp.random.normal(
                0, args.temperature, size=(
                    1,
                    3,
                ) + hyperparams.image_size).astype("float32")

            enc_z.append(cupy.asnumpy(z))
            lvar = xp.log(args.temperature)
            logpZ2.append(
                cupy.asnumpy(cf.gaussian_nll(z, 0, lvar).data)
            )

            x, blogd = decoder.reverse_step(z)
            x_img = make_uint8(x.data[0], num_bins_x)
            rev_x.append(x_img)
            print('rev_x', type(rev_x[0]))
            bk_logdet.append(cupy.asnumpy(blogd.data))
            print(type(bk_logdet[0]))

            factorized_z_distribution, fw_ldt = encoder.forward_step(x)
            fw_logdet.append(cupy.asnumpy(fw_ldt.data))

            factor_z = []
            ez = []
            nll = 0
            for (zi, mean, ln_var) in factorized_z_distribution:
                nll += cf.gaussian_nll(zi, mean, ln_var)
                factor_z.append(zi.data)
                ez.append(zi.data.reshape(-1,))
            
            ez = np.concatenate(ez)
            sec_z.append(ez.get())
            sec_pz.append(cupy.asnumpy(nll.data))
            sec_pz2.append(
                cupy.asnumpy(cf.gaussian_nll(ez, np.mean(ez), np.log(np.var(ez))).data ))

            enc_z = []
            rev_x = []
            bk_logdet = []
            logpZ2 = []
            fw_logdet = []
            sec_z = []
            sec_pz = []
            sec_pz2 = []
            np.save('sample/' + str(j) + 'enc_z.npy', enc_z)
            np.save('sample/' + str(j) + 'rev_x.npy', rev_x)
            bk_logdet = cupy.asnumpy(bk_logdet)
            np.save('sample/' + str(j) + 'bk_logdet.npy', bk_logdet)
            logpZ2 = cupy.asnumpy(logpZ2)
            np.save('sample/' + str(j) + 'logpZ2.npy', logpZ2)
            fw_logdet = cupy.asnumpy(fw_logdet)
            np.save('sample/' + str(j) + 'fw_logdet.npy', fw_logdet)
            np.save('sample/' + str(j) + 'sec_z.npy', sec_z)
            sec_pz = cupy.asnumpy(sec_pz)
            np.save('sample/' + str(j) + 'sec_pz.npy', sec_pz)
            sec_pz = cupy.asnumpy(sec_pz2)
            np.save('sample/' + str(j) + 'sec_pz.npy', sec_pz2)






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-path", "-snapshot", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gpu-device", "-gpu", type=int, default=0)
    args = parser.parse_args()
    main()
