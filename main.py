# from RatingTimeGAN import timeGAN as tg
# from RatingTimeGAN import loadRatingMatrices as lrm
# from RatingTimeGAN import brownianMotion as bm
from RatingTimeGAN import TimeGAN,BrownianMotion,getTimeIndex,RML

import numpy as np
import tensorflow as tf

from typing import List

import time as timer

tf.config.set_visible_devices([], 'GPU')
print(tf.config.experimental.get_synchronous_execution())
print(tf.config.experimental.list_physical_devices())
print(tf.config.threading.get_inter_op_parallelism_threads())
print(tf.config.threading.get_intra_op_parallelism_threads())

'Data type for computations'
# use single precision for GeForce GPUs
dtype = np.float32

# seed for reproducibility
seed = 0
tf.random.set_seed(seed)

'Parameters for Brownian motion'
# time steps of Brownian motion, has to be such that mod(N-1,12)=0
N = 5 * 12 + 1
# trajectories of Brownian motion will be equal to batch_size for training
# M = batch_size = 1
# number of independent Brownian motions, takes the role of latent dimension
n = 1
# Brownian motion class with fixed datatype
BM = BrownianMotion(dtype=dtype, seed=seed)

'Load rating matrices'
# choose between 1,3,6,12 months
times = np.array([1, 3, 6, 12])
lenSeq = times.size
T = times[-1] / 12
timeIndices = getTimeIndex(T, N, times / 12)
# relative path to rating matrices:
filePaths: List[str] = ['Data/' + 'SP_' + str(x) + '_month_small' for x in times]
# exclude default row, don't change
# excludeDefaultRow = False
# permuteTimeSeries, don't change
# permuteTimeSeries = True
# load rating matrices
RML = RML(filePaths,
              dtype=dtype)
print('Load data')
ticRML = timer.time()
RML.loadData()
ctimeRML = timer.time() - ticRML
print(f'Elapsed time for loading data {ctimeRML} s.')

'Build GAN'
# training data
rm_train = RML.tfData()
print(f'Data shape: (Data,Time Seq,From Rating*To Rating)={rm_train.shape}')
# number of ratings
Krows = RML.Krows
Kcols = RML.Kcols
# batch size
batch_size = 512

# buffer size should be greater or equal number of data,
# is only important if data doesn't fit in RAM
buffer_size = rm_train.shape[0]

dataset = tf.data.Dataset.from_tensor_slices(rm_train)
dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True).batch(batch_size)

epochs = 10
saveDir = 'RatingTimeGAN/modelParams'
tGAN = TimeGAN(lenSeq, Krows, Kcols, batch_size, dtype=dtype)
tGAN.trainTimeGAN(dataset, BM, T, N, timeIndices, epochs, loadDir=saveDir)
tGAN.save(saveDir)
samples = tGAN.sample(BM,T,N,timeIndices, 10)
print(samples.shape)
samples = np.reshape(samples, (samples.shape[0], samples.shape[1], Krows, Kcols))
print(samples.shape)
for wi in range(0, 3):
    print(f'Trajectory {wi}\n')
    for ti in range(0, samples.shape[1]):
        print(f'Time {timeIndices[ti]}')
        print(samples[wi, ti, :, :])
        print(np.sum(samples[wi, ti, :, :], axis=1))