from pymef import MefSession
import pandas as pd
from neurobox.utils import channel_sort_df
import numpy as np

class Mef3(MefSession):
    def read_ts_channel_basic_info_df(self):
        bi = super().read_ts_channel_basic_info()
        bi = pd.DataFrame(bi)
        bi = channel_sort_df(bi,'name')
        for k in ['fsamp','nsamp','ufact','unit','start_time','end_time','channel_description']:
            bi[k] = bi[k].apply(lambda x: x[0])
        for k in [ 'unit','channel_description']:
            bi[k] = bi[k].apply(lambda x: x.decode('ascii'))
        return bi

    def read_ts_channels_sample(self, channel_map, sample_map, process_n=None,transforms=None):
        data = super().read_ts_channels_sample(channel_map,sample_map,process_n)
        if transforms:
            data = [transforms(x) for x in data]
        return data