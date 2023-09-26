from pymef import MefSession
import pandas as pd
from neurobox.utils import channel_sort_df
import numpy as np
from tqdm import tqdm
import unittest
from multiprocessing import Process


class Mef3Tester(MefSession):
    def __init__(self,session_path,password):
        super(Mef3Tester, self).__init__(session_path,password)
        self.session_path = session_path
        self.password = password

    def is_readable(self):
        p = Process(target=self.read_ts_channel_basic_info)
        p.start()
        p.join()
        if p.exitcode != 0:
            return False
        else:
            return True


class Mef3(MefSession):
    def __init__(self,session_path,password,transforms=None,test_read=False):
        super(Mef3, self).__init__(session_path,password)
        self.session_path = session_path
        self.password = password
        self.transforms = transforms

        if test_read:
            if not self.is_readable():
                raise Exception("File not readable")

        self._bi = self.read_ts_channel_basic_info()
        self.fs = self._bi.iloc[0]['fsamp']

    def set_transforms(self,transforms):
        self.transforms = transforms

    def read_ts_channel_basic_info(self):
        bi = super().read_ts_channel_basic_info()
        bi = pd.DataFrame(bi)
        bi = channel_sort_df(bi,'name')
        for k in ['fsamp','nsamp','ufact','unit','start_time','end_time','channel_description']:
            bi[k] = bi[k].apply(lambda x: x[0])
        for k in [ 'unit','channel_description']:
            bi[k] = bi[k].apply(lambda x: x.decode('ascii'))
        bi['path'] = self.session_path
        bi['password'] = self.password
        return bi

    def read_ts_channels_sample(self, channel_map, sample_map, process_n=None):
        data = super().read_ts_channels_sample(channel_map,sample_map,process_n)
        if self.transforms:
            data = [self.transforms(x) for x in data]
        return data

    def read_ts_channels_uutc(self, channel_map, uutc_map, process_n=None, out_nans=True):
        data = super().read_ts_channels_uutc(channel_map,uutc_map,process_n,out_nans)
        if self.transforms:
            data = [self.transforms(x) for x in data]
        return data

    def select_channels(self,channels):
        keep = []
        for i,row in self._bi.iterrows():
            if row['name'] in channels:
                keep.append(row)
        self._bi = pd.DataFrame(keep)
        return self


    def iterchannels(self,sample_map=None):
        if sample_map is None:
            sample_map = [0,self._bi.iloc[0]['nsamp']]

        for ch in self._bi['name'].tolist():
            data = self.read_ts_channels_sample(channel_map=[ch],sample_map=sample_map)[0]
            yield ch, data

    def iter_segments_nsamp(self,sample_window):
        for ch in self._bi['name'].tolist():
            start_sample = 0
            while (start_sample + sample_window) < self._bi.iloc[0]['nsamp']:
                stop_sample = start_sample+sample_window
                data = self.read_ts_channels_sample(channel_map=[ch],sample_map=[start_sample,stop_sample])[0]
                yield ch, start_sample, stop_sample, data
                start_sample += sample_window

    def iter_segments_uutc(self,uutc_window):
        for i,row in self._bi.iterrows():
            ch = row['name']
            start_uutc = row['start_time']
            while (start_uutc + uutc_window) < row['end_time']:
                stop_uutc = start_uutc + uutc_window
                data = self.read_ts_channels_uutc(channel_map=[ch],uutc_map=[start_uutc,stop_uutc])[0]
                yield ch, start_uutc, stop_uutc, data
                start_uutc += uutc_window


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


    def read_ts_batch_uutc(self,df):
        if 'channel_name' not in df.columns:
            raise Exception("channel_name not in df")
        if 'uutc_start' not in df.columns:
            raise Exception("uutc_start not in df")
        if 'uutc_stop' not in df.columns:
            raise Exception("uutc_stop not in df")

        output = []
        for i,row in df.iterrows():
            data = self.read_ts_channels_uutc(channel_map=row['channel_name'],
                                              uutc_map=[int(row['uutc_start']),int(row['uutc_stop'])])
            output.append({'channel_name': row['channel_name'],
                           'start_sample': int(row['uutc_start']),
                           'stop_sample': int(row['uutc_stop']),
                           'data': data})

        output = pd.DataFrame(output)
        return output


    def is_readable(self):
        p = Process(target=self.read_ts_channel_basic_info)
        p.start()
        p.join()
        if p.exitcode != 0:
            return False
        else:
            return True


class Test(unittest.TestCase):
    def test_select(self):
        pth = "/home/nejedly/Desktop/sub-032_ses-001_task-rest_run-01_ieeg.mefd"
        ms = Mef3(pth,"bemena").select_channels(["B1","B2"])
        stop = 1

    def test_deploy_model(self):
        class MDL():
            def __init__(self):
                pass
            def run(self,x):
                return np.zeros_like(x)

        pth = "/home/nejedly/Desktop/sub-032_ses-001_task-rest_run-01_ieeg.mefd"
        ms = Mef3(pth,"bemena").select_channels("B1").deploy_model_uutc(model=MDL(),uutc_window=10*1000000)
        stop = 1

    def test_readable(self):
        #pth = "/home/nejedly/Desktop/sub-032_ses-001_task-rest_run-01_ieeg.mefd"
        pth = "/mnt/m/d04/eeg_data/kuna_eeg/fnusa_dataset/sub-036/ses-001/ieeg/sub-036_ses-001_task-oddball_run-01_ieeg.mefd"
        ms = Mef3(pth, "bemena").is_readable()
        stop = 1
