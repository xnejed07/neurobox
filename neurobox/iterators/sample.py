import pandas as pd
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


        return signal,self._iter.iloc[item].to_dict()


class SessionBipolarSampleIterator():
    def __init__(self,session,window,step,transforms=None):
        self.session = session
        self.window = window
        self.step = step
        self.transforms = transforms
        self.prepare()

    def prepare(self):
        bi = self.session.read_ts_channel_basic_info_df()
        bi['electrode'] = bi['name'].apply(lambda name: "".join([k for k in name if not k.isdigit()]))

        bipolars = []
        for i in range(0,len(bi)-1):
            if bi.iloc[i]['electrode'] != bi.iloc[i+1]['electrode']:
                continue

            # create bipolar only if N channels in electrode, e.g. to filter out ECG1-ECG2, ECG2-ECG3
            tmp = bi[bi['electrode']==bi.iloc[i]['electrode']]
            if len(tmp) <= 3:
                continue


            bipolars.append({'name':bi.iloc[i]['name'] + "-"+bi.iloc[i+1]['name'],
                             'fsamp':bi.iloc[i]['fsamp'],
                             'nsamp':bi.iloc[i]['nsamp'],
                             'ufact':bi.iloc[i]['ufact'],
                             'unit':bi.iloc[i]['unit'],
                             'channel_description':None,
                             })
        bi = pd.DataFrame(bipolars)

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
        signal = self.session.read_ts_channels_sample(channel_map=self._iter.iloc[item]['channel_name'].split("-"),
                                                      sample_map=[self._iter.iloc[item]['start_sample'],
                                                                  self._iter.iloc[item]['end_sample']])
        if self.transforms:
            signal = self.transforms(signal)


        return signal,self._iter.iloc[item].to_dict()