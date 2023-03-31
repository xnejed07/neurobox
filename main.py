import dataloaders.dfile as dfile
import dataloaders.mef3 as mef3
import unittest
from iterators import SessionSampleIterator

class test(unittest.TestCase):
    def test_dfile(self):
        file = "/mnt/m/d04/eeg_data/kuna_eeg/data-d_fnusa_organizace/seeg/seeg-102-200909/Easrec_sciexp-seeg102_200909-0838.d"
        D = dfile.DFile(file,header_only=True)
        for data, meta in SessionSampleIterator(session=D,window=15000,step=5000).select_channels(channel_map=['B1','C1']):
            stop = 1
        assert True

    def test_mef3(self):
        file = "/mnt/m/d04/eeg_data/kuna_eeg/fnusa_dataset/sub-102/ses-001/ieeg/sub-102_ses-001_task-rest_run-01_ieeg.mefd"
        M = mef3.Mef3(session_path=file, password='bemena')
        bi = M.read_ts_channel_basic_info_df()
        for data, meta in SessionSampleIterator(session=M,window=15000,step=5000).select_channels(channel_map=['B1','C1']):
            stop = 1
        assert True
