

import h5py
from utils import *


class H5:
    def __init__(self, path):
        self.path = path
        with h5py.File(self.path,'r') as f:
            keys = np.array(f.keys())
            self._header = np.array(f['Info'])
            self._channels = [self._header[k][0] for k in range(self._header.shape[0])]
            self._channels = [k.decode("ascii") for k in self._channels]
            self._data = f['Data'][:,:]

    def read_ts_channel_basic_info(self):
        output = []
        for ch in self._channels:
            output.append({'name':ch,
                           'fsamp':None,
                           'nsamp':self._data.shape[-1],
                           'ufact':None,
                           'unit':None,
                           'start_time':None,
                           'end_time':None,
                           'channel_description':None})
        return output

    def read_ts_channel_basic_info_df(self):
        return channel_sort_df(pd.DataFrame(self.read_ts_channel_basic_info()),'name')

    def read_ts_channels_sample(self,channel_map: list, sample_map: list):
        if isinstance(channel_map,str):
            channel_map = [channel_map]

        ch_idxs = []
        for ch in channel_map:
            if ch in self._channels:
                ch_idxs.append(self._channels.index(ch))
            else:
                raise Exception(f"Invalid channel name: {ch}")

        if len(ch_idxs) == 0:
            raise Exception("Empty channel list!")

        return self._data[ch_idxs,sample_map[0]:sample_map[1]]