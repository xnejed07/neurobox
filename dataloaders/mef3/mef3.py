from pymef import MefSession
import pandas as pd
from utils import channel_sort_df

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