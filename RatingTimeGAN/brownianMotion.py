# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 16:50:41 2022

@author: kevin
"""

import numpy as np
import tensorflow as tf

from typing import Optional, Union, Type
from numpy.typing import ArrayLike, DTypeLike


def getTimeIndex(T: float,
                 N: int,
                 t: ArrayLike,
                 endpoint: Optional[bool] = True)\
        -> ArrayLike:
    return np.floor(np.array(t) * (N - int(endpoint)) / T).astype(np.int64)

class BrownianMotion:
    def __init__(self,
                 dtype: DTypeLike = np.float64,
                 seed: Optional[int] = None)\
            -> None:
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)

    def sample(self,
               T: float,
               N: int,
               M: int,
               n: Optional[int] = 1,
               mode: Optional[Union[str, np.ndarray]] = 'all')\
            -> np.ndarray:
        """
        Samples 'n' indpendent Brownian motions with 'N' time steps
        from 0 to 'T' and 'M' paths. Shape = (Time,Samples,NumberOfBM)

        Parameters
        ----------
        T : float
            Terminal time for homogeneous time grid starting in zero.
        N : int
            Number of time steps.
        M : int
            Number of trajectories.
        n : Optional[int], optional
            Number of independent Brownian motions. The default is 1.
        mode : Optional[Union[str,np.ndarray]], optional
            Decide if you want the whole trajectory (all), only the 
            endpoint (end) or specific time indices given in a numpy array. 
            The default is 'all'.

        Returns
        -------
        Brownian motions
        """
        dW = np.sqrt(T / (N - 1)) * self.rng.standard_normal((N - 1, M, n)).astype(self.dtype)
        W = np.zeros((N, M, n), dtype=self.dtype)
        W[1:, :] = dW
        if isinstance(mode,str) and mode == 'end':
            W = np.sum(W, axis=0)
        elif isinstance(mode,str) and mode == 'all':
            W = np.cumsum(W, axis=0)
        else:
            W = np.cumsum(W, axis=0)
            W = W[mode, :, :]
        return W

    def tfW(self,
            T: float,
            N: int,
            M: int,
            n: Optional[int] = 1,
            mode: Optional[str] = 'all')\
            -> tf.Tensor:
        return tf.convert_to_tensor(self.sample(T, N, M, n=n, mode=mode))


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # import tensorflow as tf
    np.random.seed(0)
    T = 1
    N = 1000
    M = 10
    n = 1
    BM = BrownianMotion(dtype=np.float32)
    fig, ax = plt.subplots()
    W = BM.sample(T, N, M, n)
    plt.plot(np.linspace(0, T, N, endpoint=True), W[:, 2, 0])
    # plt.show(block=False)
    # fig, ax = plt.subplots()
    # W = BM.sample(T, N, M, n)
    # plt.plot(np.linspace(0, T, N, endpoint=True), W[:, 2, 0])
    # plt.show(block=False)
    # fig, ax = plt.subplots()
    # W = BM.sample(T, N, M, n)
    # plt.plot(np.linspace(0, T, N, endpoint=True), W[:, 2, 0])
    # plt.show(block=False)
    # ax.axis('off')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(True)
    # ax.spines['left'].set_visible(False)
    # ax.get_yaxis().set_ticks([])
    # plt.savefig('BrownianPath.pdf')
    # np.random.seed(0)
    print(np.squeeze(W[-1, :, :]))
    print(BM.tfW(T, N, M, n, mode=np.array([0, int(N / 2), N - 1], dtype=np.int64)))
    # plt.show()
