import pandas as pd
from neurobox.utils import channel_sort_df
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

def generate_PDM_matrix(df,feature_name,time_axis='start_sample',index_axis='channel_name',plot=False,title=None,save_plot=None,normalize=False,sort_channels=False) -> pd.DataFrame:
    PDM = pd.pivot_table(df, values=feature_name, columns=time_axis, index=index_axis)
    PDM = PDM.reset_index()

    if isinstance(sort_channels,bool) and sort_channels:
        PDM = channel_sort_df(PDM, 'channel_name')

    if isinstance(sort_channels,list):
        PDM['electrode_id'] = PDM['channel_name'].apply(lambda x: sort_channels.index(x))
        PDM = PDM.sort_values(by='electrode_id')
        PDM = PDM.drop(columns=['electrode_id'])

    PDM = PDM.set_index('channel_name', drop=True)
    if plot:
        plt.figure(dpi=300)
        data = PDM.to_numpy()
        if normalize:
            data = zscore(data,axis=-1)
        plt.imshow(data,aspect='auto')
        plt.yticks(np.arange(len(PDM)),PDM.index.tolist(),fontsize=3)
        if title is not None:
            plt.title(title)
            plt.suptitle(feature_name)
        if save_plot is not None:
            plt.savefig(save_plot)
        plt.show()
    return PDM