import os
import sys
import time
import itertools
import numpy as np
import pandas as pd
from dateutil import tz
from scipy import fftpack, stats
from itertools import chain
from datetime import datetime,timedelta
import pytz
from pytz import timezone
import calendar
from common_funcs import *

logger = logging.getLogger(__name__)

def smooth_data(data,hz):
    """
    Args: data: pd dataframe of the raw acc data
        hz: scalar, the target sampling frequency
        tz_str: timezone (str), where the study is conducted
    Return:
          t0: the first timestamp of this hour
          t_active: scalar, the duration (in seconds) when the senser is on
          stamp_hz： 1d numpy array of timestamp, e.g. [0, 100, 200, ...]
          It only contains the timestamps exisiting in the observed data (within 100 ms (assume 10 hz))
          mag_hz: 1d numpy array, the corresponding magnitude at that timestamp (if multiple measures coexist at that timestamp,
          within 100ms, take the average)
    """
    t = np.array(data["timestamp"])
    x = np.array(data["x"])
    y = np.array(data["y"])
    z = np.array(data["z"])
    mag = np.sqrt(x**2+y**2+z**2)
    t_diff = t[1:]-t[:-1]
    t_active = sum(t_diff[t_diff<5*1000])
    t_active = t_active/1000/60  ## in minute
    a = np.floor((t - min(t))/(1/hz*1000))  ## bin
    stamp_hz = np.unique(a)*(1/hz*1000)
    b = []
    for i in np.unique(a):
        index = a==i
        b.append(np.mean(mag[index]))
    mag_hz = np.array(b)
    return t[0]/1000, t_active, stamp_hz, mag_hz

def step_est(stamp_hz,mag_hz,t_active,q,c):
    """
    Args: stamp_hz, mag_hz: 1d numpy array, output from smooth_data()
          t_active: a scalar [0,60], output from smooth_data()
          q,c: scalars, parameters
    Return: The first-step estimate of step count for that hour
    """
    if np.mean(mag_hz)>8:
        g = 9.8
    else:
        g = 1
    h = max(np.percentile(mag_hz,q),c*g)
    step = 0
    current = -350
    for j in range(len(stamp_hz)):
        if(mag_hz[j]>=h and stamp_hz[j]>=current+350):
            step = step + 1
            current = stamp_hz[j]
    final_step = int(step/t_active*60)
    return final_step

def acti_dur(stamp_hz,mag_hz,hz,p,h):
    """
    Args: stamp_hz, mag_hz: 1d numpy array, output from smooth_data()
          hz,p,h: scalars, parameters
    Return: The first-step estimate of user's active duration in minute [0,60]
    """
    unit = 60*1000
    max_size = int(np.floor(unit*hz/1000))
    num = int(np.floor(stamp_hz[-1]/unit))
    ## check the active level of every minute
    eff_min = 0
    act_min = 0
    for i in range(num):
        index = (stamp_hz>=unit*i)*(stamp_hz<unit*(i+1))
        if sum(index)>p*max_size:
            eff_min = eff_min + 1
            if np.std(mag_hz[index])>h*np.mean(mag_hz[index]):
                act_min = act_min + 1
    act_min_adjust = act_min/(eff_min+0.0001)*60
    return act_min_adjust

def other_stats(mag_hz,hz):
    """
    Args: mag_hz: 1d numpy array, output from smooth_data()
          hz: scalar, parameter
    Return: 1d array of other acc statistics
    """
    if np.mean(mag_hz)>8:
        mag_hz = mag_hz/9.8
    m_mag = np.mean(mag_hz)
    sd_mag = np.std(mag_hz)
    cur_len = np.mean(abs(mag_hz[1:]-mag_hz[:-1]))
    X = fftpack.fft(mag_hz)
    amplitude_spectrum = np.abs(X)/hz
    eg = sum(amplitude_spectrum**2)*hz/len(mag_hz)**2
    entropy = stats.entropy(mag_hz)
    return np.array([m_mag,sd_mag,cur_len,eg,entropy])

def hourly_stats(stamp_hz,mag_hz,t0,t_active,hz,q,c,p,h,tz_str):
    """
    This is a function integrating all summary stats for accelerometer
    Args: stamp_hz, mag_hz: 1d numpy array, output from smooth_data()
          t_active: scalar, output from smooth_data()
          hz,q,c,p,h: scalar, parameters
          tz_str: string, parameter
    Return: 1d array of all acc statistics, together with time
    """
    steps = step_est(stamp_hz,mag_hz,t_active,q,c)
    act_mins = acti_dur(stamp_hz,mag_hz,hz,p,h)
    others = other_stats(mag_hz,hz)
    time_list = stamp2datetime(t0,tz_str)
    t = t0 - time_list[4]*60 - time_list[5]
    time_array = np.array(time_list)[:4]
    result = np.concatenate((time_array,np.array([t,t_active,steps,act_mins]),others))
    return result

def first_step_summaries(study_folder,ID,time_start,time_end,hz,q,c,p,h,tz_str):
    """
    This function is another master function, which takes study folder, ID as input,
    and returns all the first-step summary stats for the observed hours (at least 5 min
    of active time in one hour) as a 2d array
    Args: study_folder,ID are strings
          time_start, time_end are starting time and ending time of the window of interest
          time should be a list of integers with format [year, month, day, hour, minute, second]
          if time_start is None and time_end is None: then it reads all the available files
          if time_start is None and time_end is given, then it reads all the files before the given time
          if time_start is given and time_end is None, then it reads all the files after the given time
          hz,q,c,p,h: scalar, parameters
          tz_str: string, parameter
    Return: 2d array of all acc statistics, each row is one hour, each col is one feature
            and final starting and ending time
    """
    try:
        sys.stdout.write('Loading the data and calculating the first-step hourly summary stats ...' + '\n')
        file_list, stamp_start, stamp_end = read_data(ID, study_folder, "accelerometer", tz_str, time_start, time_end)
        first_step_stats = []
        for data_file in file_list:
            dest_path = study_folder + "/" + ID +  "/accelerometer/" + data_file
            hour_data = pd.read_csv(dest_path)
            t0, t_active, stamp_hz, mag_hz = smooth_data(hour_data,hz)
            if t_active>4:
                first_step_stats.append(hourly_stats(stamp_hz,mag_hz,t0,t_active,hz,q,c,p,h,tz_str))
        first_step_stats = np.array(first_step_stats)
        if first_step_stats.shape[0] < 24:
            sys.stdout.write('The accelerometer data quality is too low, the algorithm stopped.' + '\n')
            logger.warning(ID + ' : The accelerometer data quality is too low.')
        return first_step_stats, stamp_start, stamp_end
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(ID) + ': ' + str(exc_value).replace(",", ""))

def hour_range(h1,h2):
    """
    Args: h1: the hour in the center
        h2: the window size on two sides
        for example: (4,2) --> [2,3,4,5,6],   (23,1) --> [22,23,0]
    Return: a 1d numpy array of hours
    """
    if h1 + h2 > 23:
        out = np.arange(0,h1+h2-24+1)
        out = np.append(np.arange(h1-h2,24),out)
    elif h1 - h2 < 0:
        out = np.arange(h1-h2+24,24)
        out = np.append(out,np.arange(0,h1+h2+1))
    else:
        out = np.arange(h1-h2,h1+h2+1)
    return out

def check_exist(a1,a2):
    """
    check if each element in a1 is in a2, both a1 and a2 should be numpy array
    Return: a bool vector of len(a1)
    """
    b = np.zeros(len(a1))
    for i in range(len(a1)):
        if sum(a1[i]==a2)>0:
            b[i] = 1
    return np.array(b,dtype=bool)

def infer_obs(current_stats,first_step_stats,w_obs):
    """
    If the observed duration is short, use observations during the same hour of day to help
    Args: current_stats: 1d array, output from hourly_stats(), or one row in first_step_stats
          first_step_stats, 2d array, output from user_stats()
          w_obs: scalar, parameter, [1,++)
    Return: a 1d numpy array of new current_stats
    """
    hours = first_step_stats[:,3]
    index = hours==current_stats[3]
    if sum(index)<3:
        index = np.arange(first_step_stats.shape[0])
    candidates = first_step_stats[index,:]
    active_ts = candidates[:,5]
    r = np.random.randint(len(active_ts))
    ## the weight of current obs
    w = current_stats[5]*w_obs/(current_stats[5]*w_obs + active_ts[r])
    acc_stats = w*current_stats[6:] + (1-w)*candidates[r,6:]
    new_current_stats = np.concatenate((current_stats[:6],acc_stats))
    return new_current_stats

def infer_mis(current_t,tz_str,first_step_stats):
    """
    If the current hour is missing, use observations during the same hour of day to help
    extreme case of infer_obs()
    Args: current_t: scalar, timestamp
          tz_str: timezone, string
          first_step_stats, 2d array, output from user_stats()
    Return: a 1d numpy array of new current_stats
    """
    time_list = stamp2datetime(current_t,tz_str)
    hour = time_list[3]
    hours = first_step_stats[:,3]
    index = hours == hour
    if sum(index)<3:
        index = np.arange(first_step_stats.shape[0])
    candidates = first_step_stats[index,:]
    r = np.random.randint(candidates.shape[0])
    new_stats = np.concatenate((np.array(time_list)[:4],np.array([current_t,0]),candidates[r,6:]))
    return new_stats

def mi_obs(current_stats,first_step_stats,w_obs,K):
    """
    multiple imputation for the observed
    Args: same as infer_obs, K means K times multiple imputation
    Return: a 1d numpy array of new current_stats
    """
    rep_mat = []
    for i in range(K):
        rep_mat.append(infer_obs(current_stats,first_step_stats,w_obs))
    rep_mat = np.array(rep_mat)
    means = np.mean(rep_mat,axis=0)[6:]
    stds = np.std(rep_mat,axis=0)[6:]
    mi_out = np.concatenate((rep_mat[0,:6],means,stds))
    return mi_out

def mi_mis(current_t,tz_str,first_step_stats,K):
    """
    multiple imputation for the missing
    Args: same as infer_mis, K means K times multiple imputation
    Return: a 1d numpy array of new current_stats
    """
    rep_mat = []
    for i in range(K):
        rep_mat.append(infer_mis(current_t,tz_str,first_step_stats))
    rep_mat = np.array(rep_mat)
    means = np.mean(rep_mat,axis=0)[6:]
    stds = np.std(rep_mat,axis=0)[6:]
    mi_out = np.concatenate((rep_mat[0,:6],means,stds))
    return mi_out

def second_step_summaries(first_step_stats,tz_str,w_obs,K):
    """
    Go through the data, each do inference on every observed and missing hour
    Args: same as infer_obs(), infer_mis()
    Return: a 2d numpy array of new summary stats, with standard deviation stacked to the right of first_step_stats
    """
    try:
        sys.stdout.write('Inferring the second-step hourly summary stats ...' + '\n')
        second_step_stats = []
        start_t = first_step_stats[0,4]
        end_t = first_step_stats[-1,4]
        k = int((end_t - start_t)/3600 + 1)
        j = 0
        for i in range(k):
            current_t = start_t + 3600*i
            if current_t == first_step_stats[j,4]:
                second_step_stats.append(mi_obs(first_step_stats[j,:],first_step_stats,w_obs,K))
                j = j + 1
            else:
                second_step_stats.append(mi_mis(current_t,tz_str,first_step_stats,K))
        second_step_stats = np.array(second_step_stats)
        return  second_step_stats
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(ID) + ': ' + str(exc_value).replace(",", ""))

def hour2day(second_step_stats):
    """
    Convert hourly summary stats to daily stats
    Args: output from second_step_summaries()
    Return: a 2d numpy array of daily summary stats
    """
    try:
        sys.stdout.write('Converting to daily summary stats ...' + '\n')
        daily_stats = []
        if second_step_stats.shape[0]>24:
            hours = second_step_stats[:,3]
            start_index = np.where(hours==0)[0]
            end_index = np.where(hours==23)[0]
            end_index = end_index[end_index>start_index[0]]
            for i in range(len(end_index)):
                index = np.arange(start_index[i],end_index[i]+1)
                temp = second_step_stats[index,:]
                newline = np.concatenate((temp[0,np.arange(3)],np.sum(temp[:,5:],axis=0)))
                daily_stats.append(newline)
            return np.array(daily_stats)
        else:
            logger.warning("The amount of accelerometer data is less than a day.")
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(ID) + ': ' + str(exc_value).replace(",", ""))


# Main function/wrapper should take standard arguments with Beiwe names:
def acc_stats_main(study_folder, output_folder, tz_str, option, hz, q, c, p, h, w_obs, K, time_start, time_end, beiwe_id):
    """
    Args:   time_start, time_end are starting time and ending time of the window of interest
            time should be a list of integers with format [year, month, day, hour, minute, second]
            if time_start is None and time_end is None: then it reads all the available files
            if time_start is None and time_end is given, then it reads all the files before the given time
            if time_start is given and time_end is None, then it reads all the files after the given time
            beiwe_id: a list of beiwe IDs
    Return: write summary stats as csv for each user during the specified period
            and a record csv file to show which users are processed, from when to when
            and logger csv file to show warnings and bugs during the run
    """
    log_to_csv(output_folder)
    logger.info("Begin")
    ## beiwe_id should be a list of str
    if beiwe_id == None:
        beiwe_id = os.listdir(study_folder)
    ## create a record of processed user ID and starting/ending time
    record = []
    for ID in beiwe_id:
        try:
            sys.stdout.write('User: '+ ID + '\n')
            ## read and process data
            first_step_stats, stamp_start, stamp_end = first_step_summaries(study_folder,ID,time_start,time_end,hz,q,c,p,h,tz_str)
            if first_step_stats.shape[0]>=24:
                second_step_stats = second_step_summaries(first_step_stats,tz_str,w_obs,K)
                if option == 'hourly':
                    hourly_stats = pd.DataFrame(np.delete(second_step_stats, 4, 1), columns=['year','month','day','hour','sensor_active_dur',
                    'step','user_active_dur','mean_magnitude', 'sd_magnitude','curve_length','energy','entropy', 'conf_step', 'conf_user_active_dur',
                    'conf_mean_magnitude', 'conf_sd_magnitude','conf_curve_length','conf_energy','conf_entropy'])
                    write_all_summaries(ID, hourly_stats, output_folder)
                if option == 'daily':
                    daily_stats = hour2day(second_step_stats)
                    daily_stats = pd.DataFrame(daily_stats, columns=['year','month','day','sensor_active_dur','step','user_active_dur',
                    'mean_magnitude', 'sd_magnitude','curve_length','energy','entropy', 'conf_step', 'conf_user_active_dur',
                    'conf_mean_magnitude', 'conf_sd_magnitude','conf_curve_length','conf_energy','conf_entropy'])
                    write_all_summaries(ID, daily_stats, output_folder)

                [y1,m1,d1,h1,min1,s1] = stamp2datetime(stamp_start,tz_str)
                [y2,m2,d2,h2,min2,s2] = stamp2datetime(stamp_end,tz_str)
                record.append([str(ID),stamp_start,y1,m1,d1,h1,min1,s1,stamp_end,y2,m2,d2,h2,min2,s2])
        except:
            logger.debug("There is a problem with respect to user %s." % str(ID))
    logger.info("End")
    ## generate the record file together with logger and comm_logs.csv
    record = pd.DataFrame(np.array(record), columns=['ID','start_stamp','start_year','start_month','start_day','start_hour','start_min','start_sec',
                          'end_stamp','end_year','end_month','end_day','end_hour','end_min','end_sec'])
    record.to_csv(output_folder + "/record.csv",index=False)
    temp = pd.read_csv(output_folder + "/log.csv")
    if temp.shape[0]==3:
      print("Finished without any warning messages.")
    else:
      print("Finished. Please check log.csv for warning messages.")


study_folder = "F:/DATA/hope"
output_folder = "C:/Users/glius/Downloads/hope_acc"
option = "daily"
beiwe_id = ['67zqexic','1rkt87d2','ychlwvnz']
tz_str = "America/New_York"
hz = 10
q = 85
c = 1.1
p = 0.1
h = 0.08
w_obs = 5
K = 10
time_start = None
time_end = None
acc_stats_main(study_folder, output_folder, tz_str, option, hz, q, c, p, h, w_obs, K, time_start, time_end, beiwe_id)
