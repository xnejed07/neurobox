import unittest
import numpy as np
import pandas as pd
from tqdm import tqdm

class ArrayFile():
    def __init__(self,data,fs,channels=None,transforms=None,uutc_start=0):
        self.data = data
        self.fs = fs

        self.time = uutc_start + 1000000*np.arange(data.shape[1])/self.fs
        self.transforms = transforms

        self.channels = channels
        if self.channels is None:
            self.channels = []
            for i in range(data.shape[0]):
                self.channels.append(f"CH_{str(i).zfill(3)}")

        self._bi = self.read_ts_channel_basic_info()



    def read_ts_channels_sample(self, channel_map=None, sample_map=None):
        if channel_map is None:
            channel_map = self.channels

        if sample_map is None:
            sample_map = [0,self.data.shape[-1]]

        if self.transforms:
            data = [self.transforms(self.data[self.channels.index(ch),sample_map[0]:sample_map[1]]) for ch in channel_map]
            return data
        else:
            data = [self.data[self.channels.index(ch),sample_map[0]:sample_map[1]] for ch in channel_map]
            return data


    def read_ts_channels_uutc(self, channel_map=None, uutc_map=None):
        if channel_map is None:
            channel_map = self.channels

        if uutc_map is None:
            uutc_map = [0,self.time[-1]]

        sample_map = [np.where(self.time >= uutc_map[0])[0][0],
                      np.where(self.time <= uutc_map[1])[0][-1]]
        return self.read_ts_channels_sample(channel_map,sample_map)

    def read_ts_channel_basic_info(self):
        bi = []
        for k in self.channels:
            bi.append({
                        'name':k,
                        'fsamp':self.fs,
                        'nsamp':self.data.shape[1],
                        'ufact':None,
                        'unit':None,
                        'start_time':self.time[0],
                        'end_time':self.time[-1],
                        'channel_description':None})
        bi = pd.DataFrame(bi)
        return bi


    def iter_segments_uutc(self,uutc_window):
        for i,row in self._bi.iterrows():
            ch = row['name']
            start_uutc = row['start_time']
            while (start_uutc + uutc_window) < row['end_time']:
                stop_uutc = start_uutc + uutc_window
                data = self.read_ts_channels_uutc(channel_map=[ch],uutc_map=[start_uutc,stop_uutc])[0]
                yield ch, start_uutc, stop_uutc, data
                start_uutc += uutc_window


    def iter_segments_samples(self,sample_window):
        for ch in self._bi['name'].tolist():
            start_sample = 0
            while (start_sample + sample_window) < self._bi.iloc[0]['nsamp']:
                stop_sample = start_sample+sample_window
                data = self.read_ts_channels_sample(channel_map=[ch],sample_map=[start_sample,stop_sample])[0]
                yield ch, start_sample, stop_sample, data
                start_sample += sample_window


    def deploy_model_uutc(self,model,uutc_window):
        output = []
        for ch, start_sample, stop_sample, data in tqdm(self.iter_segments_uutc(uutc_window)):
            y = model.run(data)
            output.append({'channel_name':ch,
                           'start_time': start_sample,
                           'stop_time':stop_sample,
                           'data':y})

        output = pd.DataFrame(output)
        return output

    def deploy_model_samples(self,model,samples_window):
        output = []
        for ch, start_sample, stop_sample, data in tqdm(self.iter_segments_samples(samples_window)):
            y = model.run(data)
            output.append({'channel_name':ch,
                           'start_sample': start_sample,
                           'stop_sample':stop_sample,
                           'data':y})

        output = pd.DataFrame(output)
        return output




class Test(unittest.TestCase):
    def test_0(self):
        data = np.random.randn(16,5000*60*60)
        af = ArrayFile(data,fs=5000)
        for (ch, start_uutc, stop_uutc, data) in af.iter_segments_samples(sample_window=16384):
            stop = 1
        stop = 1