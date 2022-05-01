# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 17:04:54 2022

@author: kevin
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path

from typing import List, Optional
from numpy.typing import ArrayLike, DTypeLike


class RML:
    def __init__(self,
                 filePaths: List[str],
                 excludeDefaultRow: Optional[bool] = False,
                 permuteTimeSeries: Optional[bool] = True,
                 vectorizeRatings: Optional[bool] = True,
                 dtype: DTypeLike = np.float64) \
            -> None:
        self.data: np.ndarray = np.array([],dtype=dtype)
        self.filePaths: List[str] = filePaths
        self.lenTimeSequence: int = len(filePaths)
        self.excludeDefaultRow: bool = excludeDefaultRow
        self.permuteTimeSeries: bool = permuteTimeSeries
        self.vectorizeRatings: bool = vectorizeRatings
        self.dtype = dtype
        self.ratings: List[str] = []
        self.Kcols : int = -1
        self.Krows: int = -1

    def loadData(self) \
            -> None:
        data: List[np.ndarray] = []
        for filePath in self.filePaths:
            path = Path.cwd() / filePath
            files = path.glob('*.csv')

            temp: List[np.ndarray] = []
            for f in files:
                df = pd.read_csv(f.absolute(), index_col=0, delimiter=';')
                temp.append(df.to_numpy().astype(self.dtype))

            tempArray = np.array(temp,dtype=self.dtype)
            if self.excludeDefaultRow:
                tempArray = tempArray[:, :-1, :]
            if df.index.name == '%':
                tempArray /= 100
            data.append(tempArray)
        self.ratings: List[str] = df.columns.tolist()
        colLen = len(self.ratings)
        rowLen = colLen - int(self.excludeDefaultRow)
        self.Krows = rowLen
        self.Kcols = colLen
        if self.permuteTimeSeries and self.lenTimeSequence > 1:
            numEntries = []
            for i in range(0, self.lenTimeSequence):
                numEntries.append(np.arange(0, data[i].shape[0], dtype=np.int64))
            permuList = np.meshgrid(*numEntries, indexing='ij')
            permArray = permuList[0].ravel().reshape(-1, 1)
            for i in range(1, self.lenTimeSequence):
                permArray = np.concatenate((permArray, permuList[i].ravel().reshape(-1, 1)), axis=1)
            self.data = np.zeros((permArray.shape[0], self.lenTimeSequence, rowLen, colLen),dtype=self.dtype)
            for i in range(0, permArray.shape[0]):
                currSeq = []
                for j in range(0, permArray.shape[1]):
                    currSeq.append(data[j][permArray[i, j], :, :])
                self.data[i, :, :, :] = np.array(currSeq)
            # self.data = self.data.transpose((0, 2, 3, 1))
        elif self.lenTimeSequence > 1:
            raise ValueError('Not yet implemented please set permuteTimeSeries = True')
        else:
            self.data = data[0]
            # add a new axis for sequence index <-> "batch size, timesteps, features"
            self.data = self.data[:, np.newaxis, :, :]
        if self.vectorizeRatings:
            self.data = np.reshape(self.data,(self.data.shape[0],self.data.shape[1],self.data.shape[2]*self.data.shape[3]))
    def tfData(self) \
            -> tf.Tensor:
        return tf.convert_to_tensor(self.data)


if __name__ == '__main__':
    import time as timer

    filePaths = ['SP_' + str(x) + '_month_small' for x in [1, 3]]
    rml = RML(filePaths, excludeDefaultRow=True, dtype=np.float32)
    print(rml.filePaths)
    tic = timer.time()
    rml.loadData()
    ctime = timer.time() - tic
    print(f'Elapsed time {ctime} s')
    print(rml.data.dtype)
    print(rml.data.shape)
    # print(rml.ratings)
    print(rml.tfData().dtype)
