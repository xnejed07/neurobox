import glob
import unittest
import pathlib
import pandas as pd




class BIDSPath:
    def __init__(self,path):
        self.path = pathlib.Path(path)
        self.name = self.path.name

        tmp = [i for i,k in enumerate(self.path.parts) if 'sub' in k]
        self.dataset = self.path.parts[tmp[0]-1]
        self.sub = self.path.parts[tmp[0]]
        self.suffix = self.path.suffix
        tmp = self.name.replace(self.suffix,"").split("_")
        self.ses = tmp[1]
        self.task = tmp[2]
        self.run = tmp[3]
        stop = 1

    def __str__(self):
        return str(self.path)

    def __repr__(self):
        return str(self.path)

    def todict(self):
        return {"path":str(self),
                "dataset":self.dataset,
                "sub":self.sub,
                "ses":self.ses,
                "task":self.task,
                "run":self.run,
                "suffix":self.suffix}

    @staticmethod
    def glob(search_path):
        files = glob.glob(search_path)
        files = [BIDSPath(f).todict() for f in files]
        files = pd.DataFrame(files)
        return files





class Test(unittest.TestCase):
    def test_0(self):
        pth = "/mnt/m/d04/eeg_data/kuna_eeg/fnusa_dataset/sub-047/ses-001/ieeg/sub-047_ses-001_task-rest_run-01_ieeg.mefd"
        dst = BIDSPath(pth)

        d = dst.todict()
        stop =1

    def test_1(self):
        files = BIDSPath.glob("/mnt/m/d04/eeg_data/kuna_eeg/fnusa_dataset/*/*/*/*.mefd")
        stop = 1
