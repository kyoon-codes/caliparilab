#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 11:08:53 2024

@author: kristineyoon
"""

Data_df=pd.DataFrame(columns=Column_names)
    data=tdt.read_block(date_dir)

    #first pre-process the photometry data
    try:
        sampling_rate = data.streams._490R.fs #acquisition rate of fluorescence data
        dLight_raw = data.streams._490R.data[int(sampling_rate):]
        Iso_raw = data.streams._405R.data[int(sampling_rate):]    
        Iso_raw=Iso_raw[:len(dLight_raw)]
        fs=sampling_rate
        time_seconds = np.linspace(int(sampling_rate),len(data.streams._490R.data)/sampling_rate, len(data.streams._490R.data)-int(sampling_rate))
    except:
        sampling_rate = data.streams[signal_name].fs #acquisition rate of fluorescence data
        dLight_raw = data.streams[signal_name].data[int(sampling_rate):]
        Iso_raw = data.streams[control_name].data[int(sampling_rate):]
        fs=sampling_rate
        time_seconds_sig = np.linspace(int(sampling_rate),len(data.streams[signal_name].data)/sampling_rate, len(data.streams[signal_name].data)-int(sampling_rate))
        time_seconds_con = np.linspace(int(sampling_rate),len(data.streams[control_name].data)/sampling_rate, len(data.streams[control_name].data)-int(sampling_rate))
        
        if len(time_seconds_sig)>len(time_seconds_con):
            dLight_raw = dLight_raw[:len(time_seconds_con)]
            Iso_raw = Iso_raw[:len(time_seconds_con)]
            time_seconds = time_seconds_con
        elif len(time_seconds_sig)<len(time_seconds_con):
            dLight_raw = dLight_raw[:len(time_seconds_sig)]
            Iso_raw = Iso_raw[:len(time_seconds_sig)]
            time_seconds = time_seconds_sig
        else:
            time_seconds=time_seconds_con
            
        #########################################################################
        #only for 2024/01/19 when I had to use the FibPho2 instead of FibPho1
        # sampling_rate = data.streams._465C.fs #acquisition rate of fluorescence data
        # dLight_raw = data.streams._465C.data[int(sampling_rate):]
        # Iso_raw = data.streams._405C.data[int(sampling_rate):]
        # fs=sampling_rate
        # time_seconds = np.linspace(int(sampling_rate),len(data.streams._465C.data)/sampling_rate, len(data.streams._465C.data)-int(sampling_rate))
        #########################################################################
    
    ##################
    # Denoising
    ##################
    # Lowpass filter - zero phase filtering (with filtfilt) is used to avoid distorting the signal.
    b,a = butter(2, 10, btype='low', fs=sampling_rate)
    dLight_denoised = filtfilt(b,a, dLight_raw)
    Iso_denoised = filtfilt(b,a, Iso_raw)
    ##################
    # Correct photobleach
    ##################  
    # The double exponential curve we are going to fit.
    def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
        '''Compute a double exponential function with constant offset.
        Parameters:
        t       : Time vector in seconds.
        const   : Amplitude of the constant offset. 
        amp_fast: Amplitude of the fast component.  
        amp_slow: Amplitude of the slow component.  
        tau_slow: Time constant of slow component in seconds.
        tau_multiplier: Time constant of fast component relative to slow. 
        '''
        tau_fast = tau_slow*tau_multiplier
        return const+amp_slow*np.exp(-t/tau_slow)+amp_fast*np.exp(-t/tau_fast)
    
    # Fit curve to dLight signal.
    max_sig = np.max(dLight_denoised)
    inital_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0      , 0      , 0      , 600  , 0],
              [max_sig, max_sig, max_sig, 36000, 1])
    dLight_parms, parm_cov = curve_fit(double_exponential, time_seconds, dLight_denoised, 
                                      p0=inital_params, bounds=bounds, maxfev=3000)
    dLight_expfit = double_exponential(time_seconds, *dLight_parms)
    
    # Fit curve to TdTomato signal.
    max_sig = np.max(Iso_denoised)
    inital_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1]
    bounds = ([0      , 0      , 0      , 600  , 0],
              [max_sig, max_sig, max_sig, 36000, 1])
    Iso_parms, parm_cov = curve_fit(double_exponential, time_seconds, Iso_denoised, 
                                      p0=inital_params, bounds=bounds, maxfev=3000)
    Iso_expfit = double_exponential(time_seconds, *Iso_parms) 
    
    ########################################################################
    #Option 1: denoise-fit exp and subtract from each(bleach)-polyfit control to signal and remove (mvt)-divide by signal_expfit (scale)
    ########################################################################
    #Subtract exponential fit from data
    dLight_detrended = dLight_denoised - dLight_expfit
    Iso_detrended = Iso_denoised - Iso_expfit
    #########################
    # Movement correction
    #########################
    slope, intercept, r_value, p_value, std_err = linregress(x=Iso_detrended, y=dLight_detrended)  
    dLight_est_motion = intercept + slope * Iso_detrended
    dLight_corrected = dLight_detrended - dLight_est_motion
    #################
    # Compute dff
    #################
    dLight_dF_F = 100*dLight_corrected/dLight_expfit
    
    #######################################################################
    # create a master time that subsamples at 20Hz
    #######################################################################
    try:
        stopTime_in_seconds=len(data.streams._490R.data)/sampling_rate
        Time = np.arange(1,stopTime_in_seconds, 0.05)      
    except:
        stopTime_in_seconds=len(data.streams[signal_name].data)/sampling_rate
        Time = np.arange(1,stopTime_in_seconds, 0.05)  
    # Time=Time[:-1000] # can use that to truncate a recoring when it crashed
    Data_df['Time']=Time
    #######################################################################
    # subsample dFF and video frames based on master Time
    #######################################################################
    Video_index_cam1=[]
    cam1_times=data.epocs.Cam1.onset
    cam1_idx=data.epocs.Cam1.data
    for frame_time in Time:
        print(frame_time)
        match_idx = np.argmin(abs(cam1_times-frame_time))
        Video_index_cam1.append(cam1_idx[match_idx])
    
    # Video_index_cam2=[]
    # cam2_times=data.epocs.Cam2.onset
    # cam2_idx=data.epocs.Cam2.data
    # for frame_time in Time:
    #     print(frame_time)
    #     match_idx = np.argmin(abs(cam2_times-frame_time))
    #     Video_index_cam2.append(cam2_idx[match_idx])
            
    Fluorescence_index=[]
    for frame_time in Time:
        print(frame_time)
        fluo_indexed_vid_frame_time=int(frame_time*fs)
        Fluorescence_index.append(fluo_indexed_vid_frame_time)
        
    #Add data into dataframe  
    Data_df['Video_index_TopView']=Video_index_cam1
    # Data_df['Video_index_SideView']=Video_index_cam1
    try:
        Data_df['Sig_470']=data.streams._490R.data[Fluorescence_index]
        Data_df['Sig_415']=data.streams._405R.data[Fluorescence_index]
        processed_dff_idx=[x-int(sampling_rate) for x in Fluorescence_index]
        Data_df['Jan2024_dff']=dLight_dF_F[processed_dff_idx]
    except:
        Data_df['Sig_470']=data.streams[signal_name].data[Fluorescence_index]
        Data_df['Sig_415']=data.streams[control_name].data[Fluorescence_index]
        processed_dff_idx=[x-int(sampling_rate) for x in Fluorescence_index]
        Data_df['Jan2024_dff']=dLight_dF_F[processed_dff_idx]
        #########################################################################
        #only for 2024/01/19 when I had to use the FibPho2 instead of FibPho1
        # Data_df['Sig_470']=data.streams._465C.data[Fluorescence_index]
        # Data_df['Sig_415']=data.streams._405C.data[Fluorescence_index]
        # processed_dff_idx=[x-int(sampling_rate) for x in Fluorescence_index]
        # Data_df['Jan2024_dff']=dLight_dF_F[processed_dff_idx]
        #########################################################################
