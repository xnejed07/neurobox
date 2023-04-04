import pandas as pd
import numpy as np

def channel_sort_df(df, chan_name, other_chan_names=[], rename_channels=False):
    """
    Function to ad 0s to channels and sort them. Meant for FNUSA

    Parameters:
    -----------
    df - dataframe\n
    chan_name - channel name column (str), sorted by this column\n
    other_chan_names - list of strings with othe channel name columns

    Returns:
    --------
    mod_df - modified and sorted dataframe\n
    """

    temp_df = df[[chan_name] + other_chan_names].copy()

    # Rename channels add 0s - for good ordering
    for channel in temp_df[chan_name].unique():
        digits = [x for x in channel if x.isdigit()]
        if len(digits) == 1:
            dig_idx = channel.index(digits[0])
            mod_chan = channel[:dig_idx] + '0' + channel[dig_idx:]
            temp_df.loc[temp_df[chan_name] == channel, chan_name] = mod_chan
            if rename_channels:
                df.loc[df[chan_name] == channel, chan_name] = mod_chan
            if other_chan_names == []:
                continue
            for other_chan in other_chan_names:
                temp_df.loc[temp_df[other_chan] == channel, other_chan] = mod_chan
                if rename_channels:
                    df.loc[df[other_chan] == channel, other_chan] = mod_chan

    temp_df.sort_values(by=chan_name, inplace=True)
    temp_df.reset_index(inplace=True)

    df = df.iloc[temp_df['index']]
    df.reset_index(inplace=True, drop=True)

    return df


def channel_sort_list(channels):
    """
    Function to ad 0s to channels and sort them. Meant for FNUSA

    Parameters:
    -----------
    channels - list\n

    Returns:
    --------
    mod_df - modified and sorted dataframe\n
    """

    # Rename channels add 0s - for good ordering
    modified_channels = []
    for channel in channels:
        digits = [x for x in channel if x.isdigit()]
        if len(digits) == 1:
            dig_idx = channel.index(digits[0])
            mod_chan = channel[:dig_idx] + '0' + channel[dig_idx:]
            modified_channels.append(mod_chan)
        else:
            modified_channels.append(channel)

    modified_channels.sort()

    for ci, channel in enumerate(modified_channels):
        digits = [x for x in channel if x.isdigit()]
        if not len(digits):
            continue
        if digits[0] == '0':
            dig_idx = channel.index(digits[0])
            modified_channels[ci] = modified_channels[ci][0:dig_idx] + modified_channels[ci][dig_idx + 1:]

    return modified_channels