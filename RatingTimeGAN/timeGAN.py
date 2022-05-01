# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 17:28:08 2022

@author: kevin
this code is an adaption of
https://github.com/ydataai/ydata-synthetic/blob/dev/src/ydata_synthetic/synthesizers/timeseries/timegan/model.py
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Sequential, Input
from tensorflow.keras.layers import GRU, LSTM, Dense
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adam
# from tensorflow import function, GradientTape, sqrt, abs, reduce_mean, ones_like, zeros_like, convert_to_tensor,float32

import time as timer

from tqdm import tqdm, trange

import brownianMotion as bm
import loadRatingMatrices as lrm

from pathlib import Path
import shutil
from datetime import datetime

from typing import List, Optional
from numpy.typing import ArrayLike, DTypeLike

# force tensorflow to use CPU, has to be on start-up
tf.config.set_visible_devices([], 'GPU')
print(tf.config.experimental.get_synchronous_execution())
print(tf.config.experimental.list_physical_devices())
print(tf.config.threading.get_inter_op_parallelism_threads())
print(tf.config.threading.get_intra_op_parallelism_threads())
# tf.config.threading.set_inter_op_parallelism_threads(2)
# tf.config.threading.set_intra_op_parallelism_threads(6)
# logs = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
#                                                       histogram_freq = 1,
#                                                       profile_batch = '500,510')
class Generator(Model):
    def __init__(self,
                 hiddenDims: List[int],
                 outputDims: int,
                 RNNLayer : Optional[tf.keras.layers.Layer] = GRU) \
            -> None:
        self.hiddenDims = hiddenDims
        self.outputDims = outputDims
        self.RNNLayer = RNNLayer

    def build(self) \
            -> Model:
        model = Sequential(name='Generator')
        for i, hD in enumerate(self.hiddenDims):
            model.add(self.RNNLayer(units=hD,
                          return_sequences=True,
                          name=f'Generator_{self.RNNLayer.__name__}_{i + 1}'))
        model.add(Dense(units=self.outputDims,
                        activation='sigmoid',
                        name='Generator_OUT'))
        return model


class Discriminator(Model):
    def __init__(self,
                 hiddenDims: List[int],
                 RNNLayer : Optional[tf.keras.layers.Layer] = GRU):
        self.hiddenDims = hiddenDims
        self.RNNLayer = RNNLayer

    def build(self):
        model = Sequential(name='Discriminator')
        for i, hD in enumerate(self.hiddenDims):
            model.add(GRU(units=hD,
                          return_sequences=True,
                          name=f'Disciminator_{self.RNNLayer.__name__}_{i + 1}'))
        model.add(Dense(units=1,
                        activation='sigmoid',
                        name='Disciminator_OUT'))
        return model


class Recovery(Model):
    def __init__(self,
                 hiddenDims: List[int],
                 featureDims: int,
                 RNNLayer : Optional[tf.keras.layers.Layer] = GRU) \
            -> None:
        self.hiddenDims = hiddenDims
        self.featureDims = featureDims
        self.RNNLayer = RNNLayer

    def build(self) \
            -> Model:
        model = Sequential(name='Recovery')
        for i, hD in enumerate(self.hiddenDims):
            model.add(GRU(units=hD,
                          return_sequences=True,
                          name=f'Recovery_{self.RNNLayer.__name__}_{i + 1}'))
        model.add(Dense(units=self.featureDims,
                        activation='sigmoid',
                        name='Recovery_OUT'))
        return model


class Embedder(Model):
    def __init__(self,
                 hiddenDims: List[int],
                 outputDims: int,
                 RNNLayer : Optional[tf.keras.layers.Layer] = GRU) \
            -> None:
        self.hiddenDims = hiddenDims
        self.outputDims = outputDims
        self.RNNLayer = RNNLayer

    def build(self) \
            -> Model:
        model = Sequential(name='Embedder')
        for i, hD in enumerate(self.hiddenDims):
            model.add(GRU(units=hD,
                          return_sequences=True,
                          name=f'Embedder_{self.RNNLayer.__name__}_{i + 1}'))
        model.add(Dense(units=self.outputDims,
                        activation='sigmoid',
                        name='Embedder_OUT'))
        return model


class Supervisor(Model):
    def __init__(self,
                 hiddenDims: List[int],
                 outputDims: int,
                 RNNLayer : Optional[tf.keras.layers.Layer] = GRU) \
            -> None:
        self.hiddenDims = hiddenDims
        self.outputDims = outputDims
        self.RNNLayer = RNNLayer

    def build(self) \
            -> Model:
        model = Sequential(name='Supervisor')
        for i, hD in enumerate(self.hiddenDims):
            model.add(GRU(units=hD,
                          return_sequences=True,
                          name=f'Supervisor_GRU_{i + 1}'))
        model.add(Dense(units=self.outputDims,
                        activation='sigmoid',
                        name='Supervisor_OUT'))
        return model


class TimeGAN:
    def __init__(self,
                lenSeq: int,
                Krows: int,
                Kcols: int,
                batch_size: int,
                dtype: DTypeLike = np.float32):

        self.lenSeq = lenSeq
        self.Krows = Krows
        self.Kcols = Kcols
        self.batch_size = batch_size
        self.dtype = dtype
        'Input placeholders'
        # Placeholder for real data
        X = Input(shape=[lenSeq, Krows * Kcols], batch_size=batch_size, name='RealData')
        # Placeholder for noise
        Z = Input(shape=[lenSeq, 1], batch_size=batch_size, name='BM')

        # Network compatibility:
        # X -> embedder -> supervisor -> recovery -> X (Supervised Autoencoder)
        # X -> embedder -> recovery -> X (Unsupervised Autoencoder)
        # X -> embedder -> discriminator -> 1 (Embedded Discriminator)
        # Z -> generatorEmbedded -> supervisor -> discriminator -> 1 (Supervised GAN)
        # Z -> generatorEmbedded -> recovery -> X (TimeGAN generator)
        self.embedder = Embedder([3, 2, 3], lenSeq).build()
        self.recovery = Recovery([3, 2, 3], Krows * Kcols).build()
        self.supervisor = Supervisor([lenSeq, lenSeq], lenSeq).build()
        self.embeddedGenerator = Generator([lenSeq, lenSeq, lenSeq], lenSeq).build()
        self.discriminator = Discriminator([lenSeq, lenSeq, lenSeq]).build()

        # Autoencoder for parameter reduction
        H = self.embedder(X)
        X_tilde = self.recovery(H)
        self.autoencoder = Model(inputs=X,
                                 outputs=X_tilde,
                                 name='Autoencoder')

        # Supervised GAN
        E_hat = self.embeddedGenerator(Z)
        H_hat = self.supervisor(E_hat)
        Y_fake = self.discriminator(H_hat)
        self.supervisedGAN = Model(inputs=Z,
                                   outputs=Y_fake,
                                   name='supervisedGAN')

        # Adversarial architecture in latent space
        Y_fake_e = self.discriminator(E_hat)
        self.embeddedGAN = Model(inputs=Z,
                                 outputs=Y_fake_e,
                                 name='embeddedGAN')

        # Synthetic data generation
        X_hat = self.recovery(H_hat)
        self.generator = Model(inputs=Z,
                               outputs=X_hat,
                               name='TimeGANgenerator')

        # Final discriminator model
        Y_real = self.discriminator(H)
        self.discriminator_model = Model(inputs=X,
                                         outputs=Y_real,
                                         name='RealDiscriminator')

        self.mse = MeanSquaredError()
        self.bce = BinaryCrossentropy()
        self.gamma = 1
        self.epochs = 10
        self._isloaded = False

    @tf.function
    def train_autoencoder(self, x, opt):
        with tf.GradientTape() as tape:
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self.mse(x, x_tilde)
            e_loss_0 = 10 * tf.sqrt(embedding_loss_t0)

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss_0, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(embedding_loss_t0)


    @tf.function
    def train_supervisor(self, x, opt):
        with tf.GradientTape() as tape:
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self.mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

        var_list = self.supervisor.trainable_variables + self.generator.trainable_variables
        gradients = tape.gradient(generator_loss_supervised, var_list)
        apply_grads = [(grad, var) for (grad, var) in zip(gradients, var_list) if grad is not None]
        opt.apply_gradients(apply_grads)
        return generator_loss_supervised

    @tf.function
    def train_embedder(self, x, opt):
        with tf.GradientTape() as tape:
            # Supervised Loss
            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self.mse(h[:, 1:, :], h_hat_supervised[:, :-1, :])

            # Reconstruction Loss
            x_tilde = self.autoencoder(x)
            embedding_loss_t0 = self.mse(x, x_tilde)
            e_loss = 10.0 * tf.sqrt(embedding_loss_t0) + 0.1 * generator_loss_supervised

        var_list = self.embedder.trainable_variables + self.recovery.trainable_variables
        gradients = tape.gradient(e_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return tf.sqrt(embedding_loss_t0)

    def discriminatorLoss(self, x,z):
        y_real = self.discriminator_model(x)
        discriminator_loss_real = self.bce(y_true=tf.ones_like(y_real),
                                      y_pred=y_real)

        y_fake = self.supervisedGAN(z)
        discriminator_loss_fake = self.bce(y_true=tf.zeros_like(y_fake),
                                      y_pred=y_real)

        y_fake_e = self.embeddedGAN(z)
        discriminator_loss_fake_e = self.bce(y_true=tf.zeros_like(y_fake_e),
                                        y_pred=y_fake_e)

        return (discriminator_loss_real +
                discriminator_loss_fake +
                self.gamma * discriminator_loss_fake_e)

    @staticmethod
    def calc_generator_moments_loss(y_true, y_pred):
        y_true_mean, y_true_var = tf.nn.moments(x=y_true, axes=[0])
        y_pred_mean, y_pred_var = tf.nn.moments(x=y_pred, axes=[0])
        g_loss_mean = tf.reduce_mean(tf.abs(y_true_mean - y_pred_mean))
        g_loss_var = tf.reduce_mean(tf.abs(tf.sqrt(y_true_var + 1.0e-6) - tf.sqrt(y_pred_var + 1.0e-6)))
        return g_loss_mean + g_loss_var

    @tf.function
    def train_generator(self, x, z, opt):
        with tf.GradientTape() as tape:
            y_fake = self.supervisedGAN(z)
            generator_loss_unsupervised = self.bce(y_true=tf.ones_like(y_fake),
                                                   y_pred=y_fake)

            y_fake_e = self.embeddedGAN(z)
            generator_loss_unsupervised_e = self.bce(y_true=tf.ones_like(y_fake_e),
                                                y_pred=y_fake_e)

            h = self.embedder(x)
            h_hat_supervised = self.supervisor(h)
            generator_loss_supervised = self.mse(h[:,1:,:],h_hat_supervised[:,:-1,:])

            x_hat = self.generator(z)
            generator_moment_loss = TimeGAN.calc_generator_moments_loss(x,x_hat)

            generator_loss = (generator_loss_unsupervised +
                              generator_loss_unsupervised_e +
                              100.0 * tf.sqrt(generator_loss_supervised) +
                              100.0 * generator_moment_loss)
        var_list = self.embeddedGenerator.trainable_variables + self.supervisor.trainable_variables
        gradients = tape.gradient(generator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))
        return generator_loss_unsupervised, generator_loss_supervised, generator_moment_loss

    @tf.function
    def train_discriminator(self, x, z, opt):
        with tf.GradientTape() as tape:
            discriminator_loss = self.discriminatorLoss(x,z)

        var_list = self.discriminator.trainable_variables
        gradients = tape.gradient(discriminator_loss, var_list)
        opt.apply_gradients(zip(gradients, var_list))

        return discriminator_loss

    def trainTimeGAN(self,dataset,BM,T,N,timeIndices,epochs,loadDir: Optional[str] = ''):
        self.epochs = epochs
        if loadDir != '':
            if self.load(loadDir):
                return

        autoencoder_opt = Adam(1e-4)
        supervisor_opt = Adam(1e-4)
        generator_opt = Adam(1e-4)
        embedder_opt = Adam(1e-4)
        discriminator_opt = Adam(1e-4)

        # train embedder

        print('\ntrain autoencoder')
        for _ in tqdm(range(epochs), desc='Autoencoder network training'):
            for step, X_ in enumerate(dataset):
                step_e_loss_t0 = self.train_autoencoder(X_, autoencoder_opt)


        # train supervisor
        print('\ntrain supervisor')
        for _ in tqdm(range(epochs), desc='Supervised network training'):
            for step, X_ in enumerate(dataset):
                step_e_loss_t0 = self.train_supervisor(X_, supervisor_opt)

        # joint training

        step_g_loss_u = step_g_loss_s = step_g_loss_v = step_e_loss_t0 = step_d_loss = 0
        print('\ntrain joint model')
        for _ in tqdm(range(epochs), desc='Joint network training'):
            currK = 1
            for step, X_ in enumerate(dataset):
                Z_ = tf.transpose(tf.reshape(BM.tfW(T, N, self.batch_size, 1, mode=timeIndices),
                                             [len(timeIndices), self.batch_size]))
                if currK <= 2:

                    #   Train generator

                    step_g_loss_u, step_g_loss_s, step_g_loss_v = self.train_generator(X_, Z_, generator_opt)

                    #   Train embedder

                    step_e_loss_t0 = self.train_embedder(X_, embedder_opt)
                    currK += 1
                else:
                    step_d_loss = self.discriminatorLoss(X_, Z_)
                    if step_d_loss > 0.15:
                        step_d_loss = self.train_discriminator(X_, Z_, discriminator_opt)
                    currK = 1

    def sample(self,n_batches: int):
        steps = n_batches // self.batch_size + 1
        data = []
        for _ in trange(steps, desc='Synthetic data generation'):
            Z_ = tf.transpose(tf.reshape(BM.tfW(T, N, self.batch_size, 1, mode=timeIndices),
                                         [len(timeIndices), self.batch_size]))
            records = self.generator(Z_)
            data.append(records)
        return np.array(np.vstack(data))

    @property
    def paramString(self)\
            -> str:
        return 'AE{0:d}_G{1:d}_lenSeq{2:d}_batch{3:d}_epochs{4:d}'.format(self.autoencoder.count_params(),
                                                              self.generator.count_params(),
                                                              self.lenSeq,
                                                              self.batch_size,
                                                              self.epochs)

    def save(self,
             saveDir : str)\
        -> None:
        if not self._isloaded:
            savePath=Path.cwd() / Path(saveDir +'/' + self.paramString)
            if savePath.exists():
                shutil.rmtree(savePath)
            savePath.mkdir(parents=True, exist_ok=True)
            self.generator.save(str(savePath)+'/'+'generator')
            self.autoencoder.save(str(savePath) + '/' + 'autoencoder')
            self.recovery.save(str(savePath) + '/' + 'recovery')
            self.embedder.save(str(savePath) + '/' + 'embedder')
            self.supervisor.save(str(savePath) + '/' + 'supervisor')
            self.discriminator.save(str(savePath) + '/' + 'discriminator')
            self.discriminator_model.save(str(savePath) + '/' + 'discriminator_model')
            self.embeddedGAN.save(str(savePath) + '/' + 'embeddedGAN')
            self.embeddedGenerator.save(str(savePath) + '/' + 'embeddedGenerator')

    def load(self,
             loadDir : str)\
            -> bool:
        loadPath = Path.cwd() / Path(loadDir + '/' + self.paramString)
        if loadPath.exists():
            self.generator = tf.keras.models.load_model(str(loadPath)+'/'+'generator')
            self.autoencoder = tf.keras.models.load_model(str(loadPath) + '/' + 'autoencoder')
            self.embedder = tf.keras.models.load_model(str(loadPath) + '/' + 'embedder')
            self.supervisor = tf.keras.models.load_model(str(loadPath) + '/' + 'supervisor')
            self.discriminator = tf.keras.models.load_model(str(loadPath) + '/' + 'discriminator')
            self.discriminator_model = tf.keras.models.load_model(str(loadPath) + '/' + 'discriminator_model')
            self.embeddedGAN = tf.keras.models.load_model(str(loadPath) + '/' + 'embeddedGAN')
            self.embeddedGenerator.save(str(loadPath) + '/' + 'embeddedGenerator')
            self._isloaded = True
            return True
        else:
            return False



if __name__ == '__main__':
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
    BM = bm.BrownianMotion(dtype=dtype,seed=seed)

    'Load rating matrices'
    # choose between 1,3,6,12 months
    times = np.array([1, 3, 6, 12])
    lenSeq = times.size
    T = times[-1] / 12
    timeIndices = bm.getTimeIndex(T, N, times / 12)
    # relative path to rating matrices:
    filePaths: List[str] = ['SP_' + str(x) + '_month_small' for x in times]
    # exclude default row, don't change
    # excludeDefaultRow = False
    # permuteTimeSeries, don't change
    # permuteTimeSeries = True
    # load rating matrices
    RML = lrm.RML(filePaths,
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

    epochs=10
    saveDir = 'modelParams'
    tGAN=TimeGAN(lenSeq, Krows, Kcols, batch_size, dtype=dtype)
    tGAN.trainTimeGAN(dataset,BM,T,N,timeIndices,epochs,loadDir = saveDir)
    tGAN.save(saveDir)
    samples=tGAN.sample(10)
    print(samples.shape)
    samples=np.reshape(samples,(samples.shape[0],samples.shape[1],Krows,Kcols))
    print(samples.shape)
    for wi in range(0,3):
        print(f'Trajectory {wi}\n')
        for ti in range(0,samples.shape[1]):
            print(f'Time {timeIndices[ti]}')
            print(samples[wi,ti,:,:])
            print(np.sum(samples[wi,ti,:,:],axis=1))
