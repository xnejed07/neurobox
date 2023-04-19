from pymef import MefSession
import pandas as pd
from neurobox.utils import channel_sort_df
import numpy as np
from tqdm import tqdm

class Mef3RAM(MefSession):
    def __init__(self,session_path,password):
        super(Mef3RAM, self).__init__(session_path=session_path,password=password)
        bi = self.read_ts_channel_basic_info_df()
        self._channels = [k['name'] for _,k in bi.iterrows()]
        self._data = {}
        print("Preloading Mef3 data file to RAM:")
        for ch in tqdm(self._channels):
            self._data[ch] = super().read_ts_channels_sample(channel_map=[ch],sample_map=[0,bi.iloc[0]['nsamp']])[0]


    def read_ts_channel_basic_info_df(self):
        bi = super().read_ts_channel_basic_info()
        bi = pd.DataFrame(bi)
        bi = channel_sort_df(bi,'name')
        for k in ['fsamp','nsamp','ufact','unit','start_time','end_time','channel_description']:
            bi[k] = bi[k].apply(lambda x: x[0])
        for k in [ 'unit','channel_description']:
            bi[k] = bi[k].apply(lambda x: x.decode('ascii'))
        return bi

    def read_ts_channels_sample(self, channel_map, sample_map,transforms=None):
        data = [self._data[k][sample_map[0]:sample_map[1]] for k in channel_map]
        if transforms:
            data = [transforms(x) for x in data]
        return data