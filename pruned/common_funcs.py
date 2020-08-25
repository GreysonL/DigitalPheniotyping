import os
import sys
import pandas as pd
import numpy as np
from dateutil import tz
from datetime import datetime
import pytz
from pytz import timezone
import calendar
import logging

logger = logging.getLogger(__name__)

def setup_csv(name, directory, header):
    '''
    Creates a csv file with the given column labels.
    Overwrites a file with the same name.

    Args:
        name (str):  Name of csv file to create.
        directory (str):  Path to location for csv file.
        header (list):  List of column headers (str).

    Returns:
        path (str): Path to the new csv file.
    '''
    path = os.path.join(directory, name + '.csv')
    if os.path.exists(path):
        logger.warning('Overwriting existing file with that name.')
    f = open(path, 'w')
    f.write(','.join(header) + '\n')
    f.close()
    return(path)


# Dictionary of log record attributes:
# (For details:  https://docs.python.org/3.8/library/logging.html?highlight=logging#logrecord-attributes)
log_attributes = {
'asctime,msecs': '%(asctime)s', # Human-readable time with milliseconds.
'created': '%(created)f',       # Unix timestamp (seconds since epoch).
'filename': '%(filename)s',     # Filename portion of pathname.
'funcName': '%(funcName)s',     # Originating function.
'levelname': '%(levelname)s', # Message level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'.
'levelno': '%(levelno)s',     # Numeric message level.
'lineno': '%(lineno)d',       # Source line number, if available.
'message': '%(message)s',     # Logged message.
'module': '%(module)s',       # Module name.
'msecs': '%(msecs)d',         # Millisecond portion of timestamp.
'name': '%(name)s',              # Name of originating logger.
'pathname': '%(pathname)s',      # Path to originating file.
'process': '%(process)d',        # Process id, if available.
'processName': '%(processName)s',# Process name, if available.
'relativeCreated': '%(relativeCreated)d', # Milliseconds since logging was loaded.
'thread': '%(thread)d',          # Thread id, if available.
'threadName': '%(threadName)s',  # Thread name, if available.
 }

def attributes_to_csv(attribute_list):
    '''
    Given a list of attributes (keys of log_attributes), returns a logging
    format with header for writing records to CSV.
    '''
    try:
        log_format = type('logging_format', (), {})()
        attributes = [log_attributes[a] for a in attribute_list]
        log_format.attributes = ','.join(attributes)
        log_format.header = []
        for a in attribute_list:
            if ',' in a: log_format.header += a.split(',') # hack for asctime
            else: log_format.header.append(a)
    except:
        logger.warning('Unable to assemble logging format.')
    return(log_format)

# Simple format for logging messages:
basic_format = attributes_to_csv([
    'created',
    'asctime,msecs',
    'levelname',
    'module',
    'funcName',
    'message',
    ])

# More comprehensive format for logging messages, including traceback info:
traceback_format = attributes_to_csv([
    'created',
    'asctime,msecs',
    'levelname',
    'module',
    'funcName',
    'message',
    'lineno',
    'pathname'
    ])

def log_to_csv(log_dir, level = logging.DEBUG,
               log_name = 'log',
               log_format = traceback_format.attributes,
               log_header = traceback_format.header):
    '''
    Configure the logging system to write messages to a csv.
    Overwrites any existing logging handlers and configurations.

    Args:
        log_dir (str): Path to a directory where log messages should be written.
        level (int):  An integer between 0 and 50.
            Set level = logging.DEBUG to log all messages.
        log_name (str): Name for the log file.
        log_format (str): The format argument for logging.basicConfig.
            For available attributes and formatting instructions, see:
            https://docs.python.org/3.8/library/logging.html?highlight=logging#logrecord-attributes)
        log_header (list): Header for the csv.

    Returns:
        None
    '''
    try:
        # initialize csv
        filepath = setup_csv(name = log_name, directory = log_dir, header = log_header)
        # configure logging output
        logging.basicConfig(format = log_format,
                            filename = filepath,
                            level = level, force = True)
        # success message
        logger.info('Writing log messages to %s.csv...' % log_name)
    except:
        logger.warning('Unable to write logging messages.')


def datetime2stamp(time_list,tz_str):
    """
    Docstring
    Args: time_list: a list of integers [year, month, day, hour (0-23), min, sec],
          tz_str: timezone (str), where the study is conducted
    please use
    ## from pytz import all_timezones
    ## all_timezones
    to check all timezones
    Return: Unix time, which is what Beiwe uses
    """
    try:
        loc_tz =  timezone(tz_str)
        loc_dt = loc_tz.localize(datetime(time_list[0], time_list[1], time_list[2], time_list[3], time_list[4], time_list[5]))
        utc = timezone("UTC")
        utc_dt = loc_dt.astimezone(utc)
        timestamp = calendar.timegm(utc_dt.timetuple())
        return timestamp
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(exc_value).replace(",", ""))


def stamp2datetime(stamp,tz_str):
    """
    Docstring
    Args: stamp: Unix time, integer, the timestamp in Beiwe
          tz_str: timezone (str), where the study is conducted
    please use
    ## from pytz import all_timezones
    ## all_timezones
    to check all timezones
    Return: a list of integers [year, month, day, hour (0-23), min, sec] in the specified tz
    """
    try:
        loc_tz =  timezone(tz_str)
        utc = timezone("UTC")
        utc_dt = utc.localize(datetime.utcfromtimestamp(stamp))
        loc_dt = utc_dt.astimezone(loc_tz)
        return [loc_dt.year, loc_dt.month,loc_dt.day,loc_dt.hour,loc_dt.minute,loc_dt.second]
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(exc_value).replace(",", ""))

def filename2stamp(filename):
    """
    Docstring
    Args: filename (str), the filename of communication log
    Return: UNIX time (int)
    """
    try:
        [d_str,h_str] = filename.split(' ')
        [y,m,d] = np.array(d_str.split('-'),dtype=int)
        h = int(h_str.split('_')[0])
        stamp = datetime2stamp((y,m,d,h,0,0),'UTC')
        return stamp
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(exc_value).replace(",", ""))

def read_data(ID:str, study_folder: str, datastream:str, tz_str: str, time_start, time_end):
    """
    Docstring
    Args: ID: beiwe ID; study_folder: the path of the folder which contains all the users
          datastream: 'gps','accelerometer','texts' or 'calls'
          tz_str: where the study is/was conducted
          starting time and ending time of the window of interest
          time should be a list of integers with format [year, month, day, hour, minute, second]
          if time_start is None and time_end is None: then it reads all the available files
          if time_start is None and time_end is given, then it reads all the files before the given time
          if time_start is given and time_end is None, then it reads all the files after the given time
    return: a panda dataframe of the datastream (not for accelerometer data!) and corresponding starting/ending timestamp (UTC),
            you can convert it to numpy array as needed
            For accelerometer data, intsead of a panda dataframe, it returns a list of filenames
            The reason is the volumn of accelerometer data is too large, we need to process it on the fly:
            read one csv file, process one, not wait until all the csv's are imported (that may be too large in memory!)
    """
    df = pd.DataFrame()
    stamp_start = 0 ; stamp_end = 0
    folder_path = study_folder + "/" + ID +  "/" + str(datastream)
    ## if text folder exists, call folder must exists
    try:
        if not os.path.exists(study_folder + "/" + ID):
            logger.warning('User '+ str(ID) + ' does not exist, please check the ID again.')
        elif not os.path.exists(folder_path):
            logger.warning('User '+ str(ID) + ' : ' + str(datastream) + ' data are not collected.')
        else:
            filenames = np.array(os.listdir(folder_path))
            ## create a list to convert all filenames to UNIX time
            filestamps = np.array([filename2stamp(filename) for filename in filenames])
            ## find the timestamp in the identifier (when the user was enrolled)
            identifier_Files = os.listdir(study_folder + "/" + ID + "/identifiers")
            identifiers = pd.read_csv(study_folder + "/" + ID + "/identifiers/"+ identifier_Files[0], sep = ",")
            ## now determine the starting and ending time according to the Docstring
            stamp_start1= identifiers["timestamp"][0]/1000
            if time_start == None:
                stamp_start = stamp_start1
            else:
                stamp_start2 = datetime2stamp(time_start,tz_str)
                stamp_start = max(stamp_start1,stamp_start2)
            ##Last hour: look at all the subject's directories (except survey) and find the latest date for each directory
            directories = os.listdir(study_folder + "/" + ID)
            directories = list(set(directories)-set(["survey_answers","survey_timings"]))
            lastDate = []
            for i in directories:
                files = os.listdir(study_folder + "/" + ID + "/" + i)
                lastDate.append(files[-1])
            stamp_end_vec = [filename2stamp(j) for j in lastDate]
            stamp_end1 = max(stamp_end_vec)
            if time_end == None:
                stamp_end = stamp_end1
            else:
                stamp_end2 =  datetime2stamp(time_end,tz_str)
                stamp_end = min(stamp_end1,stamp_end2)
            ## extract the filenames in range
            files_in_range = filenames[(filestamps>=stamp_start)*(filestamps<stamp_end)]
            if len(files_in_range) == 0:
                logger.warning('User '+ str(ID) + ' : There are no ' + str(datastream) + ' data in range.')
            else:
                if datastream!='accelerometer':
                    ## read in the data one by one file and stack them
                    for data_file in files_in_range:
                        dest_path = folder_path + "/" + data_file
                        hour_data = pd.read_csv(dest_path)
                        if df.shape[0]==0:
                            df = hour_data
                        else:
                            df = df.append(hour_data,ignore_index=True)
                    sys.stdout.write("Data imported ..." + '\n')
                    return df, stamp_start, stamp_end
                else:
                    return files_in_range, stamp_start, stamp_end
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(ID) + ': ' + str(exc_value).replace(",", ""))


def write_all_summaries(ID, stats_pdframe, output_folder):
    """
    Docstring
    Args: ID: str, stats_pdframe is pd dataframe (summary stats)
          output_path should be the folder path where you want to save the output
    Return: write out as csv files named by user ID
    """
    try:
        if os.path.exists(output_folder)==False:
            os.mkdir(output_folder)
        stats_pdframe.to_csv(output_folder + "/" + str(ID) +  ".csv",index=False)
        print('Done.')
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.debug(str(ID) + ': ' + str(exc_value).replace(",", ""))
