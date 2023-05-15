from pymef import MefSession
import pandas as pd
from neurobox.utils import channel_sort_df
import numpy as np
from tqdm import tqdm

class Mef3(MefSession):
    def __init__(self,session_path,password,transforms=None):
        super(Mef3, self).__init__(session_path,password)
        self.session_path = session_path
        self.transforms = transforms
        self._bi = self.read_ts_channel_basic_info_df()

    def read_ts_channel_basic_info_df(self):
        bi = super().read_ts_channel_basic_info()
        bi = pd.DataFrame(bi)
        bi = channel_sort_df(bi,'name')
        for k in ['fsamp','nsamp','ufact','unit','start_time','end_time','channel_description']:
            bi[k] = bi[k].apply(lambda x: x[0])
        for k in [ 'unit','channel_description']:
            bi[k] = bi[k].apply(lambda x: x.decode('ascii'))
        return bi

    def read_ts_channels_sample(self, channel_map, sample_map, process_n=None):
        data = super().read_ts_channels_sample(channel_map,sample_map,process_n)
        if self.transforms:
            data = [self.transforms(x) for x in data]
        return data

    def iterchannels(self,sample_map=None):
        if sample_map is None:
            sample_map = [0,self._bi.iloc[0]['nsamp']]

        for ch in self._bi['name'].tolist():
            data = self.read_ts_channels_sample(channel_map=[ch],sample_map=sample_map)[0]
            yield ch, data

    def itersegments(self,sample_window):
        for ch in self._bi['name'].tolist():
            start_sample = 0
            while (start_sample + sample_window) < self._bi.iloc[0]['nsamp']:
                stop_sample = start_sample+sample_window
                data = self.read_ts_channels_sample(channel_map=[ch],sample_map=[start_sample,stop_sample])[0]
                yield ch, start_sample, stop_sample, data
                start_sample += sample_window


    def deploy_model(self,model,sample_window):
        output = []
        for ch, start_sample, stop_sample, data in tqdm(self.itersegments(sample_window)):
            y = model.run(data)
            output.append({'channel_name':ch,
                           'start_sample': start_sample,
                           'stop_sample':stop_sample,
                           'data':y})

        output = pd.DataFrame(output)
        return output

