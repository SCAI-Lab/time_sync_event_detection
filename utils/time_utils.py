import sys
import os

from datetime import datetime, timedelta


# convert the frame rate to seconds
def frame2sec(frame_cur, frame_begin, FS):
    return (frame_cur - frame_begin) / FS

# convert seconds to datetime object
def sec2datetime(second, start_time):
    return start_time + timedelta(seconds=second)


# convert the datetime object to unix epoch time
def time2UnixEpochTime(datetime_obj):
    # Define the format
    datetime_format = '%Y-%m-%d %H:%M:%S'

    unix_timestamp = int(datetime_obj.timestamp())
    
    return unix_timestamp

def unixEpochTime2time(u_time):
    return datetime.fromtimestamp(u_time)