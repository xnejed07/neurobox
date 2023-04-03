from dataloaders import *
from utils import *


class SessionSampleIterator():
    def __init__(self,session,window,step,transforms=None):
        self.session = session
        self.window = window
        self.step = step
        self.transforms = transforms
        self.prepare()

    def prepare(self):
        bi = self.session.read_ts_channel_basic_info_df()

        self._iter = []
        for i,row in bi.iterrows():
            T0 = 0
            while (T0 + self.window) < row['nsamp']:
                self._iter.append({'channel_name':row['name'],
                                  'start_sample':T0,
                                  'end_sample':T0+self.window})
                T0 += self.step
        self._iter = pd.DataFrame(self._iter)
        self._iter['path'] = self.session.path
        stop = 1

    def select_channels(self,channel_map):
        self._iter = self._iter[self._iter.apply(lambda x: True if x['channel_name'] in channel_map else False, axis=1)]
        self._iter = self._iter.reset_index(drop=True)
        return self



    def __len__(self):
        return len(self._iter)

    def __getitem__(self, item):
        signal = self.session.read_ts_channels_sample(channel_map=self._iter.iloc[item]['channel_name'],
                                                      sample_map=[self._iter.iloc[item]['start_sample'],
                                                                  self._iter.iloc[item]['end_sample']])
        if self.transforms:
            signal = self.transforms(signal)


        return signal,self._iter.iloc[item]