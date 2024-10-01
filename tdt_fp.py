import pandas as pd
import numpy as np
import math
import os
import pickle
import sys
import tdt
import matplotlib


sys.path.append('/Volumes/Kristine/D2_Ethanol_FiPho/4832')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# in data:  epocs: the TTL inputs (Po1, Po4, Po6, Po7, Cam1, Cam2, PABB)
#           snips: empty
#           streams: DelF, _490R, _405R, Fi1r
#           scalars: Fi1i
#           info [struct]
#           time_ranges: array([[ 0.], [inf]])

Home_dir = '/Volumes/Kristine/D2_Ethanol_FiPho/'
mice = ['4832']
#Done:
#Need to do:
for mouse in mice:
    mouse_dir = os.path.join(Home_dir, mouse)
    Dates = [x for x in os.listdir(mouse_dir) if x.isnumeric()]
    for date in Dates:
        date_dir = os.path.join(mouse_dir, date)
        Files_in_date_dir = os.listdir(mouse_dir)
        File_types = [x[-3:] for x in Files_in_date_dir]
        if 'pkl' in File_types:
            file_to_open = os.path.join(mouse_dir, date, 'TDTlog_' + mouse + '_' + date + '.pkl')
            with open(file_to_open, 'rb') as f:
                Data_df = pickle.load(f, encoding='latin1')
            if 'Video_index_TopView' in list(Data_df):
                print('TDTlog_' + mouse + '_' + date + '.pkl already exists - skipping')
                continue

        Column_names = ['Mouse', 'Date', 'Time', 'LeverPress', 'Lick', 'Reinforcer',
                        'HouseLight', 'EMAGlow', 'EMAGhigh', 'Sig_470', 'Sig_415',
                        'Task', 'dFF', 'Video_index_TopView', 'Video_index_SideView']

        Data_df = pd.DataFrame(columns=Column_names)
        data = tdt.read_block(date_dir)
        fs = data.streams._490R.fs

        if len(data.epocs.Cam1.data) > len(data.epocs.Cam2.data):
            Data_df['Time'] = data.epocs.Cam2.onset[
                              :-10]  # removing the last ones guaranties that we dont count an extra frame in the case that we stoppped the recorsing midframe
            Data_df['Video_index_TopView'] = data.epocs.Cam2.data[:-10]
            Video_index_cam1 = []
            Fluorescence_index = []
            Data_df['Sig_470'] = data.streams._490R.data[Fluorescence_index]
            Data_df['Sig_415'] = data.streams._405R.data[Fluorescence_index]
            Data_df['dFF'] = data.streams.DelF.data[Fluorescence_index]
            for frame_time in data.epocs.Cam2.onset[:-10]:
                # Other video indexing
                mask = frame_time < data.epocs.Cam1.onset[:-10]
                vid_idx = next((int(data.epocs.Cam1.data[i]) for i, x in enumerate(mask) if x),
                               None)  # need to take the index in the data.epocs.Cam1.data because enumerating straight would not account for the drop indices of droped frames
                Video_index_cam1.append(vid_idx)
                # Fluorescence data indexing (ie. get the closest capture to the recorded reference video frame)
                fluo_indexed_vid_frame_time = int(frame_time * fs)
                Fluorescence_index.append(fluo_indexed_vid_frame_time)
            # Add data into dataframe
            Data_df['Video_index_SideView'] = Video_index_cam1
            Data_df['Sig_470'] = data.streams._490R.data[Fluorescence_index]
            Data_df['Sig_415'] = data.streams._405R.data[Fluorescence_index]
            Data_df['dFF'] = data.streams.DelF.data[Fluorescence_index]
        else:
            Data_df['Time'] = data.epocs.Cam1.onset[:-10]
            Data_df['Video_index_SideView'] = data.epocs.Cam1.data[:-10]
            Video_index_cam2 = []
            Fluorescence_index = []
            for frame_time in data.epocs.Cam1.onset[:-10]:
                # Other video indexing
                mask = frame_time < data.epocs.Cam2.onset[:-10]
                vid_idx = next((int(data.epocs.Cam2.data[i]) for i, x in enumerate(mask) if x), None)
                Video_index_cam2.append(vid_idx)
                # Fluorescence data indexing (ie. get the closest capture to the recorded reference video frame)
                fluo_indexed_vid_frame_time = int(frame_time * fs)
                Fluorescence_index.append(fluo_indexed_vid_frame_time)
            # Add data into dataframe
            Data_df['Video_index_TopView'] = Video_index_cam2
            Data_df['Sig_470'] = data.streams._490R.data[Fluorescence_index]
            Data_df['Sig_415'] = data.streams._405R.data[Fluorescence_index]
            Data_df['dFF'] = data.streams.DelF.data[Fluorescence_index]
        # Add basic info
        Data_df.at[:, 'Mouse'] = mouse
        Data_df.at[:, 'Date'] = date

        Names = ['LeverPress', 'Lick', 'Reinforcer', 'HouseLight', 'EMAGlow', 'EMAGhigh']
        TDT_ports = ['Po0_', 'Po1_', 'Po2_', 'Po6_', 'Po3_', 'Po5_']
        for a, b in zip(Names, TDT_ports):
            try:
                timestamps_ON = data.epocs[b].onset
                int_indices_ON = []
                for ts_ON in timestamps_ON:
                    if math.isinf(ts_ON):
                        continue
                    mask = ts_ON < Data_df['Time'].values
                    behavON_idx = next((i for i, x in enumerate(mask) if x), None)
                    if not math.isinf(behavON_idx):
                        int_indices_ON.append(behavON_idx)
                timestamps_OFF = data.epocs[b].offset
                int_indices_OFF = []
                for ts_OFF in timestamps_OFF:
                    if math.isinf(ts_OFF):
                        continue
                    mask = ts_OFF < Data_df['Time'].values
                    if ts_OFF>Data_df['Time'].values[-1]: #deals with licks that occur right at the end of recording
                        continue
                    behavOFF_idx = next((i for i, x in enumerate(mask) if x), None)
                    if not math.isinf(behavOFF_idx):
                        int_indices_OFF.append(behavOFF_idx)
                TTL_values = np.zeros_like(Data_df['Time'].values)
                if sum([y - x for x, y in zip(int_indices_ON, int_indices_OFF)]) > len(
                        int_indices_ON):  # in case some event were faster than one sec (ie. LPs, licks)
                    for c, d in zip(int_indices_ON, int_indices_OFF):
                        TTL_values[c:d] = 1
                else:
                    for c in int_indices_ON:
                        TTL_values[c] = 1

                Data_df[a] = TTL_values
            except AttributeError:
                print('Oops! ' + b + ' is empty.')
                TTL_values = np.zeros_like(Data_df['Time'].values)
                Data_df[a] = TTL_values
                continue
        Data_df.to_pickle(os.path.join(date_dir, 'TDTlog_' + mouse + '_' + date + '.pkl'), protocol=4)
        print('TDTlog_' + mouse + '_' + date + '.pkl was saved')

Home_dir = '/Volumes/SAMSUNG USB/TDT Fiber Photometry'
mice = ['4030'] #, '4038', '4041'
for mouse in mice:
    mouse_dir = os.path.join(Home_dir, mouse)
    Dates = [x for x in os.listdir(mouse_dir) if x.isnumeric()]
    for date in Dates:
        date_dir = os.path.join(mouse_dir, date)
        file_to_open = os.path.join(date_dir, 'TDTlog_' + mouse + '_' + date + '.pkl')
        with open(file_to_open, 'rb') as f:
            Data_df = pickle.load(f, encoding='latin1')
            Manual_dFF = (Data_df.loc[:, 'Sig_470'] - Data_df.loc[:, 'Sig_415']) / Data_df.loc[:, 'Sig_415']
            Data_df['Manual_dFF'] = Manual_dFF
            Data_df.to_pickle(os.path.join(date_dir, 'TDTlog_' + mouse + '_' + date + '.pkl'), protocol=4)
            print('TDTlog_' + mouse + '_' + date + '.pkl was saved')