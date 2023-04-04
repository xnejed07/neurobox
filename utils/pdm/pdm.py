import pandas as pd
from utils import channel_sort_df
import matplotlib.pyplot as plt
import numpy as np

def generate_PDM_matrix(df,feature_name,time_axis='start_sample',index_axis='channel_name',plot=False,title=None,save_plot=None) -> pd.DataFrame:
    PDM = pd.pivot_table(df, values=feature_name, columns=time_axis, index=index_axis)
    PDM = PDM.reset_index()
    PDM = channel_sort_df(PDM, 'channel_name')
    PDM = PDM.set_index('channel_name', drop=True)
    if plot:
        plt.figure(dpi=300)
        plt.imshow(PDM.to_numpy(),aspect='auto')
        plt.yticks(np.arange(len(PDM)),PDM.index.tolist(),fontsize=3)
        if title is not None:
            plt.title(title)
        if save_plot is not None:
            plt.savefig(save_plot)
        plt.show()
    return PDM