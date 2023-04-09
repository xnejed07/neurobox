import numpy as np
import torch
from neurobox.dataloaders.dfile import DFile
from neurobox.dataloaders.mef3 import Mef3
from neurobox.dataloaders.h5 import H5
from neurobox.pipelines import pipeline_torch,pipeline_numpy
import unittest
from neurobox.iterators import SessionSampleIterator, SessionBipolarSampleIterator
from neurobox.utils import generate_PDM_matrix
import scipy

class test(unittest.TestCase):
    def test_dfile(self):
        file = "/mnt/m/d04/eeg_data/kuna_eeg/data-d_fnusa_organizace/seeg/seeg-102-200909/Easrec_sciexp-seeg102_200909-0838.d"
        D = DFile(file,header_only=True)
        for data, meta in SessionSampleIterator(session=D,window=15000,step=5000).select_channels(channel_map=['B1','C1']):
            stop = 1
        assert True

    def test_mef3(self):
        file = "/mnt/m/d04/eeg_data/kuna_eeg/fnusa_dataset/sub-102/ses-001/ieeg/sub-102_ses-001_task-rest_run-01_ieeg.mefd"
        M = Mef3(session_path=file, password='bemena')
        bi = M.read_ts_channel_basic_info_df()
        for data, meta in SessionSampleIterator(session=M,window=15000,step=5000).select_channels(channel_map=['B1','C1']):
            stop = 1
        assert True

    def test_h5(self):
        file = "/mnt/m/d05/hfo_envelopes/sleep/seeg-102/raw_files/seeg-102_00.h5"
        H = H5(path=file)
        bi = H.read_ts_channel_basic_info_df()
        for data, meta in SessionSampleIterator(session=H,window=15000,step=5000).select_channels(channel_map=['B1','C1']):
            stop = 1

        stop = 1


    def test_pipeline(self):
        file = "/mnt/m/d05/hfo_envelopes/sleep/seeg-102/raw_files/seeg-102_00.h5"
        H = H5(path=file)

        I = SessionSampleIterator(H,window=5000,step=5000)

        class Power(torch.nn.Module):
            classNames = ['Power']
            def __init__(self):
                super().__init__()

            def forward(self,x):
                x = torch.pow(x,2)
                x = x.mean()
                return x

            def inference(self,x):
                x = torch.from_numpy(x).float()
                return self.forward(x)


        class Features(torch.nn.Module):
            classNames = ['Power1','Power2']
            def __init__(self):
                super().__init__()
                self.feature1 = Power()
                self.feature2 = Power()
            def inference(self,x):
                x = torch.from_numpy(x).float()
                return torch.stack([self.feature1(x),
                                    self.feature2(x)]).view(len(Features.classNames))




        result = pipeline_torch(I,Power())
        generate_PDM_matrix(result,'Power',plot=True)
        stop = 1

    def test_bipolar_pipeline(self):
        file = "/mnt/m/d05/hfo_envelopes/sleep/seeg-102/raw_files/seeg-102_00.h5"
        H = H5(path=file)

        I = SessionBipolarSampleIterator(H,window=5000,step=5000)

        class RelEntropy:
            classNames = ['RelEntropy']
            def __init__(self):
                pass
            def inference(self,x):
                try:
                    x = np.nan_to_num(x)
                    y = (x[0,:] - x[0,:].min())/(x[0,:].max() - x[0,:].min())
                    z = (x[1,:] - x[1,:].min())/(x[1,:].max() - x[1,:].min())
                    yp = np.histogram(y,bins=10)[0]/y.shape[0]
                    zp = np.histogram(z,bins=10)[0]/z.shape[0]

                    REN_yz = np.sum(yp * np.log(yp/zp))
                    REN_zy = np.sum(zp * np.log(zp/yp))
                    REN = 0.5*(REN_yz+REN_zy)
                    return REN.reshape(1,1)
                except Exception as exc:
                    return np.array([0]).reshape(1,1)



        result = pipeline_numpy(I,RelEntropy())
        generate_PDM_matrix(result,'RelEntropy',plot=True)
        stop = 1

