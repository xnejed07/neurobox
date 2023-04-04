import binio
import numpy as np
import pandas as pd

from neurobox.utils.helper_code import channel_sort_df


class DFile():
    def __init__(self, path, header_only=False):
        self._header = read_d_header(file_name = path)
        self._data = None
        self.path = path
        if header_only is False:
            self._preload_data()

    def _preload_data(self):
        self._data = read_d_data(sheader=self._header[0],
                                 ch_list=[i for i, ch in enumerate(self._header[1]['channel_names'])],
                                 samp_start=0,
                                 samp_end=self._header[0]['nsamp'] - 1)
        self._data = self._data.T

    def read_ts_channel_basic_info(self):
        output = []
        for ch in self._header[1]['channel_names']:
            output.append({'name':ch,
                           'fsamp':self._header[0]['fsamp'],
                           'nsamp':self._header[0]['nsamp'],
                           'ufact':None,
                           'unit':None,
                           'start_time':(self._header[1]['time_info'])*1e6,
                           'end_time':(self._header[1]['time_info'] + self._header[0]['nsamp']/self._header[0]['fsamp'])*1e6,
                           'channel_description':None})
        return output

    def read_ts_channel_basic_info_df(self):
        return channel_sort_df(pd.DataFrame(self.read_ts_channel_basic_info()),'name')


    def read_ts_channels_sample(self, channel_map: list, sample_map: list):
        if isinstance(channel_map,str):
            channel_map = [channel_map]

        ch_idxs = []
        for ch in channel_map:
            if ch in self._header[1]['channel_names']:
                ch_idxs.append(self._header[1]['channel_names'].index(ch))
            else:
                raise Exception(f"Invalid channel name: {ch}")

        if len(ch_idxs) == 0:
            raise Exception("Empty channel list!")

        if self._data is None:
            self._preload_data()

        return self._data[ch_idxs,sample_map[0]:sample_map[1]]



def samp_from_uUTC(s_header, x_header, uUTC):
    """
    Function to convert uUTC time to file samples.

    Parameters
    ----------
    s_header: dict
        .d file standard header
    x_header: dict
        .d file extended header
    uUTC: int or list
        either one time or a list

    Returns
    -------
    sample: int or list
        number of list of samples in file
    """

    # Get the file start uUTC
    file_start = x_header['time_info'] * 1e6

    # Function to create samp
    create_samp = lambda x: (float(x - file_start) / 1e6) * s_header['fsamp']

    # Check if the uUTC variable is list
    if type(uUTC) == list:
        samp = [create_samp(x) for x in uUTC]
    elif type(uUTC) == np.ndarray:
        samp = create_samp(uUTC)
    else:
        samp = create_samp(uUTC)

    return samp


def read_d_header(file_name, read_tags=False):
    """
    Function to read .d file standard and extended header

    Parameters
    ----------
    file_name: str
        name of the file
    read_tags: bool
        whether the tags should be read(default=False)

    Returns
    -------
    sheader: dict
        standard header
    xheader: dict
        extended header
    """
    # Open the .d file
    f = open(file_name, "rb")

    # Reading standard header

    sheader_struct = binio.new(
        "# Standard header struct\n"
        "15  : string    : sign\n"
        "1   : string    : ftype\n"
        "1   : uint8     : nchan\n"
        "1   : uint8     : naux\n"
        "1   : uint16    : fsamp\n"
        "1   : uint32    : nsamp\n"
        "1   : uint8     : d_val # This is to be done\n"
        "1   : uint8     : unit\n"
        "1   : uint16    : zero\n"
        "1   : uint16    : data_org\n"
        "1   : int16     : data_xhdr_org\n"
    )

    sheader = sheader_struct.read_dict(f)
    sheader['file_name'] = file_name
    # Create a dictionary in d_val field - to be done
    dval_dict = {}
    dval_dict['value'] = sheader['d_val']
    dval_dict['data_invalid'] = int(np.floor(sheader['d_val'] / 128) % 2)
    dval_dict['data_packed'] = int(np.floor(sheader['d_val'] / 64) % 2)
    dval_dict['block_structure'] = int(np.floor(sheader['d_val'] / 32) % 2)
    dval_dict['polarity'] = int(np.floor(sheader['d_val'] / 16) % 2)
    dval_dict['data_calib'] = int(np.floor(sheader['d_val'] / 8) % 2)
    dval_dict['data_modified'] = int(np.floor(sheader['d_val'] / 4) % 2)
    dval_dict['data_cell_size'] = int(sheader['d_val'] % 4)
    sheader['d_val'] = dval_dict

    # Fix the data_org_fields
    sheader['data_org'] = 16 * sheader['data_org']
    sheader['data_xhdr_org'] = 16 * sheader['data_xhdr_org']

    ### Reading extended header
    xheader = {}
    # We need to define a dictionary for xhdr (we don't have switch-case)
    xhdr_dict = {16725: [" : uint8 : authentication_key\n", 0],
                 22082: [" : uint8 : block_var_list\n", 0],
                 16707: [" : uint8 : channel_atrib\n", 0],
                 18755: [" : uint8 : calib_info\n", 0],
                 20035: [" : uint8 : channel_names\n", 0],  # Finish here
                 17988: [" : uint8 : dispose_flags\n", 0],
                 18756: [" : char  : data_info\n", 0],
                 19526: [" : char  : file_links\n", 0],
                 21318: [" : int16 : freq\n", 2],  # Finish here
                 17481: [" : uint32: patient_id\n", 1],  # Finish here
                 19024: [" : char  : project_name\n", 0],
                 16978: [" : uint8 : rblock\n", 0],
                 18003: [" : char  : source_file\n", 0],
                 17748: [" : char  : text_record\n", 0],
                 18772: [" : uint32: time_info\n", 1],  # UTC from 1.1.1970
                 21588: [" : uint16: tag_tableen\n", 2, " : uint32: tag_tableoff\n", 2],
                 22612: [" : char  : text_extrec\n", 0],
                 0: 0
                 }

    if sheader['data_xhdr_org'] != 0:
        f.seek(sheader['data_xhdr_org'])
        cont = 1
        while cont:
            field_struct = binio.new(
                "1   : uint16    : mnemo\n"
                "1   : uint16    : field_len\n"
            )
            field = field_struct.read_dict(f)

            # xhdr dictionary check
            if field['mnemo'] in xhdr_dict:

                if field['mnemo'] == 0:
                    cont = 0
                    sheader['datapos'] = f.tell()
                elif field['mnemo'] == 21588:
                    io_txt = str(xhdr_dict[field['mnemo']][1]) + xhdr_dict[field['mnemo']][0] + str(
                        xhdr_dict[field['mnemo']][3]) + xhdr_dict[field['mnemo']][2]
                    field_cont = binio.new(io_txt)
                    xheader.update(field_cont.read_dict(f))
                elif xhdr_dict[field['mnemo']][1] == 0:
                    io_txt = str(field['field_len']) + xhdr_dict[field['mnemo']][0]
                    field_cont = binio.new(io_txt)
                    xheader.update(field_cont.read_dict(f))
                else:
                    io_txt = str(xhdr_dict[field['mnemo']][1]) + xhdr_dict[field['mnemo']][0]
                    field_cont = binio.new(io_txt)
                    xheader.update(field_cont.read_dict(f))

            else:
                f.seek(field['field_len'], 1)

    # Now fix stuff

    if 'channel_names' in xheader:
        char_lst = [chr(x) for x in xheader['channel_names']]
        channels = [''.join(char_lst[x:x + 4]) for x in range(0, len(char_lst), 4)]
        channels = [x.replace('\x00', '') for x in channels]
        xheader['channel_names'] = channels
    if 'tag_tableen' in xheader:
        xheader['tag_table'] = xheader['tag_tableen'] + xheader['tag_tableoff']
        xheader.pop('tag_tableen')
        xheader.pop('tag_tableoff')

    ### Read tags
    if 'tag_table' in xheader and read_tags:
        xheader['tags'] = {}
        xheader['tags']['tag_pos'] = []
        xheader['tags']['tag_class'] = []
        xheader['tags']['tag_selected'] = []
        xheader['tags']['classes'] = []
        # Fix datasets bug in D-file with defoff and listoff
        pr, nb = get_prec(sheader)
        while xheader['tag_table'][2] < (sheader['nchan'] * sheader['nsamp'] * nb + sheader['data_org']) and \
                xheader['tag_table'][2] > sheader['datapos']:
            xheader['tag_table'][2] += 16 ** 8
            xheader['tag_table'][3] += 16 ** 8

        # Read tag list
        f.seek(xheader['tag_table'][3])
        tag_list_struct = binio.new(
            str(int(4 * np.floor(float(xheader['tag_table'][1]) / 4.0))) + "   : uint8    : tlist")
        tag_list = tag_list_struct.read_dict(f)
        tag_list = tag_list['tlist']
        tag_list = [tag_list[x:x + 4] for x in range(0, len(tag_list), 4)]
        for x in range(len(tag_list)):
            tag_list[x][2] = tag_list[x][2] % 128
        tag_pos = []
        for x in tag_list:
            tag_pos.append((x[0] * 1) + (x[1] * 256) + (x[2] * 65536))
        xheader['tags']['tag_pos'] = tag_pos
        xheader['tags']['tag_class'] = [x[3] % 128 for x in tag_list]
        xheader['tags']['tag_selected'] = [np.floor(float(x[3]) / 128.0) for x in tag_list]

        # Fix the long datasets bug in D-file (positions > 2 ** 23-1)
        if xheader['tags']['tag_pos'] != []:
            cont1 = 1
            while cont1:
                wh = [list(np.diff(xheader['tags']['tag_pos'])).index(x) for x in
                      list(np.diff(xheader['tags']['tag_pos'])) if x < 0]
                if wh == []:
                    cont1 = 0
                else:
                    xheader['tags']['tag_pos'][wh[0] + 1:] = [x + (2 ** 23) for x in
                                                              xheader['tags']['tag_pos'][wh[0] + 1:]]

        # Read the tag table
        currpos = xheader['tag_table'][2]
        cont1 = 1
        while cont1:
            f.seek(currpos)
            tag_read_dic = binio.new(
                "2  :   char    : abrv\n"
                "1  :   uint16  : n\n"
                "1  :   uint16  : txtlen\n"
                "1  :   uint16  : txtoff\n")

            curr_tag = tag_read_dic.read_dict(f)
            currpos += 8
            if np.floor(curr_tag['n'] / 32768):
                cont1 = 0
            f.seek(curr_tag['txtoff'] + xheader['tag_table'][2])
            xheader['tags']['classes'].append(
                [curr_tag['abrv'], curr_tag['n'] % 32768, str(f.read(curr_tag['txtlen']))])

    f.close()

    return sheader, xheader


def read_d_data(sheader, ch_list, samp_start, samp_end):
    """
    Function to read the data from .d file

    Parameters
    ----------

    sheader: dict
        standard header
    ch_list: list
        list of channel idxs
    samp_start: int
        start sample
    samp_end: int
        end sample

    Returns
    -------
    data: list
        data from .d file (list), corresponding to ch_list\n

    """

    ### Open the .d file
    f = open(sheader['file_name'], "rb")

    ### Retype the start and end
    samp_start = int(samp_start)
    samp_end = int(samp_end)

    ### Just a precaution
    if samp_end >= sheader['nsamp']:
        print("\n End sample " + str(samp_end) + " equal or bigger than n samples of file (" + str(
            sheader['nsamp']) + "), setting to the end of the file \n")
        samp_end = sheader['nsamp'] - 1

    ### Get the precision - we have function for this!!!
    prec, nb = get_prec(sheader)

    ### Get the first sample
    ds1 = sheader['data_org'] + samp_start * nb * sheader['nchan']
    n_samp = samp_end - samp_start
    f.seek(ds1)

    # Reading method 1) sample by sample, slow but good for memory

    ### Allocate the output
    #    out_array = np.zeros([n_samp,len(ch_list)],dtype=prec)
    #
    #    ### Read the data
    #    for samp in range(n_samp):
    #        dat_type=binio.new(str(sheader['nchan'])+" : "+prec+" : data")
    #        dd = dat_type.read_struct(f)
    #        out_array[samp,:] = np.array(dd.data)[ch_list].astype(prec)
    #
    #    f.close()
    #
    #    return out_array

    # Reading method 2) fast but blocks up the memory

    ### Read the data
    dat_type = binio.new(str(n_samp * sheader['nchan']) + " : " + prec + " : data")
    try:
        dd = dat_type.read_struct(f)
    except:
        print('Error reading data')
        return

    f.close()

    if len(ch_list) == 1:
        return np.array(dd.data).reshape(n_samp, sheader['nchan']).astype(prec)[:, ch_list[0]]
    else:
        return np.array(dd.data).reshape(n_samp, sheader['nchan']).astype(prec)[:, ch_list]

    # Reading method 3) the combination of the above


#    out_array = np.zeros([n_samp,len(ch_list)],dtype=prec)
#
#    ### Read the data
#    chunk = 60*5000
#    samp = None
#    for samp in range(int(np.floor(n_samp / chunk))):
#        dat_type=binio.new(str(chunk*sheader['nchan'])+" : "+prec+" : data")
#        dd = dat_type.read_struct(f)
#        out_array[samp*chunk:(samp+1)*chunk] = np.array(dd.data).reshape(chunk,sheader['nchan']).astype(prec)[:,ch_list]
#
#    ### Read the last piece
#    if samp == None: samp=0
#    rest_chunk_piece = n_samp % chunk
#    dat_type=binio.new(str(rest_chunk_piece*sheader['nchan'])+" : "+prec+" : data")
#    dd = dat_type.read_struct(f)
#    out_array[(samp+1)*chunk:] = np.array(dd.data).reshape(rest_chunk_piece,sheader['nchan']).astype(prec)[:,ch_list]
#
#    f.close()
#
#    return out_array


def get_prec(sheader):
    """
    Helper funciton for reading tags

    Parameters
    ----------
    sheader: dict
        standard header

    Returns:
    --------
    result: tuple
        precision ,nb
    """
    if sheader['ftype'] == u'D':
        if sheader['d_val']['data_cell_size'] == 2:
            return 'int16', 2
        elif sheader['d_val']['data_cell_size'] == 3:
            return 'int32', 4
        else:
            return 'uint8', 1
    elif sheader['ftype'] == u'R':
        return 'float32', 4
    else:
        return 'uint8', 1


def read_dmm_header(file_path):
    """
    Function to read .d-- file standard header

    Parameters
    ----------
    file_path: str
        name of the file

    Returns
    -------
    sheader: dict
        standard header
    """

    # Open the .d file
    f = open(file_path, "rb")

    # Reading standard header

    sheader_struct = binio.new(
        "# Standard header struct\n"
        "15  : string    : sign\n"
        "1   : string    : ftype\n"
        "1   : uint8     : nchan\n"
        "1   : uint8     : naux\n"
        "1   : uint16    : fsamp\n"
        "1   : uint32    : nsamp\n"
        "1   : uint8     : d_val # This is to be done\n"
        "1   : uint8     : unit\n"
        "1   : uint16    : zero\n"
        "1   : uint16    : data_org\n"
        "1   : int16     : data_xhdr_org\n"
    )

    sheader = sheader_struct.read_dict(f)
    sheader['file_name'] = file_path
    # Create a dictionary in d_val field - to be done
    dval_dict = {}
    dval_dict['value'] = sheader['d_val']
    dval_dict['data_invalid'] = int(np.floor(sheader['d_val'] / 128) % 2)
    dval_dict['data_packed'] = int(np.floor(sheader['d_val'] / 64) % 2)
    dval_dict['block_structure'] = int(np.floor(sheader['d_val'] / 32) % 2)
    dval_dict['polarity'] = int(np.floor(sheader['d_val'] / 16) % 2)
    dval_dict['data_calib'] = int(np.floor(sheader['d_val'] / 8) % 2)
    dval_dict['data_modified'] = int(np.floor(sheader['d_val'] / 4) % 2)
    dval_dict['data_cell_size'] = int(sheader['d_val'] % 4)
    sheader['d_val'] = dval_dict

    # Fix the data_org_fields
    sheader['data_org'] = 16 * sheader['data_org']
    sheader['data_xhdr_org'] = 16 * sheader['data_xhdr_org']

    f.close()

    return sheader


def read_dmm_data(file_path, sheader, ch_list, samp_start, samp_end):
    """
    Function to read the data from .d-- file

    Parameters
    ----------

    file_path: str
        path to file
    sheader: dict
        standard header
    ch_list: list
        list of channel idxs
    samp_start: int
        start sample
    samp_end: int
        end sample

    Returns
    -------
    data: list
        data from .d file (list), corresponding to ch_list
    """

    ### Open the .d file
    f = open(sheader['file_name'], "rb")

    ### Retype the start and end
    samp_start = int(samp_start)
    samp_end = int(samp_end)

    ### Just a precaution
    if samp_end >= sheader['nsamp']:
        print("\n End sample " + str(samp_end) + " equal or bigger than n samples of file (" + str(
            sheader['nsamp']) + "), setting to the end of the file \n")
        samp_end = sheader['nsamp'] - 1

    ### Get the precision - we have function for this!!!
    prec, nb = get_prec(sheader)

    ### Get the first sample
    ds1 = sheader['data_org'] + (samp_start - 1) * nb * sheader['nchan']
    n_samp = samp_end - samp_start
    f.seek(ds1)

    ### Read the data
    dat_type = binio.new(str(n_samp * sheader['nchan']) + " : " + prec + " : data")
    try:
        f.seek(ds1)
        dd = dat_type.read_struct(f)
    except:
        print('Error reading data')
        # print '\n '+str(samp_end)+' in '+str(sheader['nsamp'])+' \n'
        return

    if len(ch_list) == 1:
        return np.array(dd.data).reshape(n_samp, sheader['nchan']).astype(prec)[:, ch_list[0]]
    else:
        return np.array(dd.data).reshape(n_samp, sheader['nchan']).astype(prec)[:, ch_list]









