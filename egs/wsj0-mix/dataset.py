"""
wsj-*mix Dataset class: A dataset interface to process wsj-{2,3}mix dataset
@author : Sunit Sivasankaran, Inria-Nancy
"""

import torch
from torch.utils import data
import json
import os
import numpy as np
import soundfile as sf
from asteroid.data.wav import SingleWav


class WSJmixDataset(data.Dataset):
    """
    A interface to process the
    Args:
        wav_len_list: str. A file containing <wav_id> <sample_len>
        wav_base_path: str. Base dir path to obtain the wav files. \
                Should find mix, s1, s2 etc in this folder
        callback_func: func, A function to process raw wav file
        elements: List of elements you want to acess. Ex: mix, s1, s2 and so on
        sample_rate: int. Sampling rate of the data
        segment: Float. Length of the segments used for training, in seconds
                By default returns the full signal. If segment is set to a
                float value, signals less that segment lengths are removed.
    """
    def __init__(self, wav_len_list, wav_base_path, callback_func=None,
            elements=['mix', 's1', 's2'], sample_rate=8000, segment=None):
        self.segment = segment
        segment_samples = segment * sample_rate if segment is not None else -1
        assert os.path.exists(wav_len_list), wav_len_list+' does not exists'
        data.Dataset.__init__(self)
        id_list = []
        id_wav_map = {}
        with open(wav_len_list) as fid:
            for line in fid:
                wav_id, wav_len = line.strip().split()
                wav_len = int(wav_len)
                id_list.append(wav_id)
                if segment_samples != -1 and wav_len < segment_samples:
                    print("Drop {} utts. {} (shorter than {} samples)".format(
                        wav_id, wav_len/sample_rate, segment))
                    continue
                if wav_id not in id_wav_map:
                   id_wav_map[wav_id] = {}
                   for _ele_ in elements:
                        id_wav_map[wav_id][_ele_] = SingleWav(\
                                os.path.join(wav_base_path, _ele_, \
                                wav_id) + '.wav')
                   id_wav_map[wav_id]['sample'] = wav_len
        self.id_list = list(id_wav_map.keys()) 
        self.id_wav_map = id_wav_map
        self.len = len(id_wav_map)
        # Create an identity function if callback is None
        self.callback_func = callback_func if callback_func is not None else\
                lambda x: x
        print("{}% file dropped".format(100*(1-self.len/len(id_list))))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        raise NotImplementedError

    def shuffle_list(self):
        """
        Shuffle the id list
        """
        np.random.shuffle(self.id_list)


class WSJ2mixDataset(WSJmixDataset):
    """
    Interface to get 2 mix dataset
    Args:
        wav_len_list: str. A file containing <wav_id> <sample_len>
        wav_base_path: str. Base dir path to obtain the wav files. \
                Should find mix, s1, s2 etc in this folder
        callback_func: func, A function to process raw wav file
        sample_rate: int. Sampling rate of the data
        segment: Float. Length of the segments used for training, in seconds
                By default returns the full signal. If segment is set to a 
                float value, signals less that segment lengths are removed.
    """
    def __init__(self, wav_len_list, wav_base_path, callback_func=None, \
            sample_rate=8000, segment=None):
        self.sources = ['s1', 's2']
        WSJmixDataset.__init__(self, wav_len_list, wav_base_path,\
                elements=['mix'] + self.sources, sample_rate=sample_rate, \
                segment=segment)

    def __getitem__(self, idx):
        item_id = self.id_list[idx]
        mixture = self.id_wav_map[item_id]["mix"].\
                    random_part_data(self.segment).T[0]
        source_arrays = []
        for _src_ in self.sources:
            source_arrays.append(self.id_wav_map[item_id][_src_].\
                    random_part_data(self.segment).T[0])
        sources = torch.from_numpy(np.vstack(source_arrays))
        mixture = torch.from_numpy(mixture)
        return self.callback_func(mixture, sources) 


class WSJ3mixDataset(WSJ2mixDataset):
    """
    Interface to get 3 mix dataset
    Args:
        wav_len_list: str. A file containing <wav_id> <sample_len>
        wav_base_path: str. Base dir path to obtain the wav files. \
                Should find mix, s1, s2 etc in this folder
        callback_func: func, A function to process raw wav file
        sample_rate: int. Sampling rate of the data
        segment: Float. Length of the segments used for training, in seconds
                By default returns the full signal. If segment is set to a
                float value, signals less that segment lengths are removed.
    """
    def __init__(self, wav_len_list, wav_base_path, callback_func=None,\
            sample_rate=8000, segment=None):
        sources = ['s1', 's2', 's3']
        WSJ2mixDataset.__init__(self, wav_len_list, wav_base_path,\
                elements=['mix'] + sources, sample_rate=sample_rate, \
                segment=segment)
        self.sources = sources
