#!/usr/bin/env python
from __future__ import print_function
from builtins import input
import sys
import os
import rospy
import rosbag
import time
from scipy.io import loadmat
import numpy as np
import itertools
import argparse

import tf
import std_msgs
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage

parser = argparse.ArgumentParser(description='Creates a rosbag from autonomous driving datasets.')
parser.add_argument('dataset_dir', metavar='DATASET_DIR', type=str,
                    help='The directory containing the dataset. For example /media/Datasets/IJRR-Dataset-1')
parser.add_argument('output_bag', metavar='OUTPUT_BAG', type=str,
                    help='The path to the output bag.')
args = parser.parse_args()

dataset_path = os.path.abspath(args.dataset_dir)
output_path = os.path.abspath(args.output_bag)
BASE_FRAME_ID = "world"
GPS_FRAME_ID = "gps_odom"
IMU_FRAME_ID = "imu_odom"
LAS_FRAME_ID = "velodyne"
TF_TOPIC = "/tf"
LAS_TOPIC = "/velodyne_points"

# logged times are UNIX timestamps in microseconds
def unix_to_rostime(timestamp):
    secs = int(np.floor(timestamp * 1e-6))
    nsecs = int( ( timestamp - ( secs * 1e6 ) ) * 1e3 )
    time_ = rospy.Time()
    time_.set(secs, nsecs)
    return time_

# Using WGS85 lat long elevation to spatial x y z in meters
def latlongalt_to_XYZ(lat_lon_alt):
    # X Y Z output in meters
    # see http://www.mathworks.de/help/toolbox/aeroblks/llatoecefposition.html
    lat = lat_lon_alt[:,0]
    lon = lat_lon_alt[:,1]
    alt = lat_lon_alt[:,2]
    R = np.float64(6378137.0)        # Radius of the Earth (in meters)
    FF = np.float64((1.0 - np.float64(1.0/298.257223563))**2)  # Flattening factor WGS84 Model
    cosLat = np.cos(lat)
    sinLat = np.sin(lat)
    c      = 1/np.sqrt(cosLat**2 + FF * sinLat**2)
    s      = c * FF
    x = (R * c + alt) * cosLat * np.cos(lon)
    y = (R * c + alt) * cosLat * np.sin(lon)
    z = (R * s + alt) * sinLat
    xyz = np.zeros(lat_lon_alt.shape)
    xyz[:,0] = x
    xyz[:,1] = y
    xyz[:,2] = z
    return xyz

def latlong_to_rel_xy(lat_lon):
    #init
    lat = lat_lon[:,0]
    lon = lat_lon[:,1]
    lat_rad = lat * np.pi / 180.
    lon_rad = lon * np.pi / 180.
    dlat_rad = lat_rad - lat_rad[0]
    dlon_rad = lon_rad - lon_rad[0]
    #constants
    R = 6368494. # earth radius (m) in detroit at sea level
    r = R * np.cos(lat_rad[0]) # radius of longitude circle at init latitude
    #conversion
    dx = r * np.sin(dlon_rad)
    dy = R * np.sin(dlat_rad)
    xy = np.zeros(lat_lon.shape)
    xy[:,0] = dx
    xy[:,1] = dy
    return xy

def gps_xy_to_theta(xy):
    dx = - xy[:-1,0] + xy[1:,0]
    dy = - xy[:-1,1] + xy[1:,1]
    dd = np.sqrt(dx*dx + dy*dy)
    dd = np.concatenate( (dd[0][None], dd) )
    theta = np.arctan2(dy, dx)
    theta = np.concatenate( (theta[0][None], theta) )
    stheta = np.copy(theta)
    last_good_t = theta[0]
    for i, t in enumerate(theta):
        if dd[i] < 0.025:
            stheta[i] = last_good_t
        else:
            last_good_t = t
    return stheta

def fetch_gps(indx, *args):
    """ 
    Fetch stored observations at given index.
    
    indx:  index of the observation to fetch
    args:  iterables each of length > indx, containing observations.

    returns: observation at index indx for each iterable arg.
    """
    if True in [indx >= len(arg) for arg in args]:
        return [np.nan for arg in args]
    return [arg[next_gps_indx] for arg in args]

fetch_imu = fetch_gps

def fetch_las(indx, las_dir, las_files):
    """ Fetch next laser observation """
    next_las_file = os.path.join(las_dir, las_files[next_las_indx])
    next_las = loadmat(next_las_file)['SCAN']
    next_las_XYZ = next_las['XYZ'][0][0].T
    next_las_time = next_las['timestamp_laser'][0][0][0][0]
    if indx >= len(las_files):
        return np.nan, np.nan
    return next_las_time, next_las_XYZ


# Open GPS (Gps.mat should be saved using the load_gps matlab function)
gps_mat_path = os.path.join(dataset_path, "Gps.mat")
try:
    gps = loadmat(gps_mat_path)
except:
    print("Gps data not found. Continue without it? Y/[n]: ", end='')
    keys = input()
    if keys in ['Y', 'Yes', 'y', 'yes']:
        GPS_DISABLED = True
    else:
        quit()
else:
    GPS_DISABLED = False
    gps = gps['pose_gps']
    gps_times = gps['utime'][0][0][::2].flatten()
    gps_lat_lon_el_theta = gps['lat_lon_el_theta'][0][0][::2]
    gps_X_Y_Z = latlongalt_to_XYZ(gps_lat_lon_el_theta[:,:3])
    gps_x_y_z = np.concatenate(
            (
            latlong_to_rel_xy(gps_lat_lon_el_theta[:,:2]),
            gps_lat_lon_el_theta[:,2][:,None]
            ), axis = 1)
    gps_theta = gps_xy_to_theta(gps_x_y_z)
    gps_cov = gps['gps_cov'][0][0][::2]
# import matplotlib.pyplot as plt
# plt.ion()
# plt.figure()
# plt.plot(gps_cov / np.max(gps_cov, axis=0))
# plt.plot(gps_theta)
# plt.show()
# plt.figure()

# Open IMU Pose
imu_mat_path = os.path.join(dataset_path, "Pose-Applanix.mat")
imu = loadmat(imu_mat_path)['pose_applanix']
imu_times = imu['utime'][0][0].flatten()
imu_pos = imu['pos'][0][0]
imu_pos = imu_pos - imu_pos[0] # set 0, 0, 0 to first timestamp
imu_q = imu['orientation'][0][0]

# List velodyne laser scan files
las_dir = os.path.join(dataset_path, "SCANS")
las_files = os.listdir(las_dir)
las_files.sort()

# Initialize indices
next_gps_indx = 0
next_imu_indx = 0
next_las_indx = 0


# Fetch first observations
if GPS_DISABLED:
    next_gps_time = np.nan
    gps_times = []
else:
    next_gps_time, next_gps_x_y_z, next_gps_theta, next_gps_cov = fetch_gps(
            gps_times,
            gps_x_y_z,
            gps_theta,
            gps_cov,
    )
next_imu_time, next_imu_pos, next_imu_q = fetch_imu(
        imu_times,
        imu_pos,
        imu_q,
)
next_las_time, next_las_XYZ = fetch_las(next_las_indx, las_dir, las_files)

total_messages = len(gps_times) + len(imu_times) + len(las_files)

if os.path.isfile(output_path):
    print("File {} already exists, replace it? [Y]/n: ".format(output_path), end='')
    keys = input()
    if keys in ['', 'Y', 'Yes', 'y', 'yes']:
        os.remove(output_path)
    else:
        quit()
try:
    bag = rosbag.Bag(output_path, 'w')
    for count in itertools.count():
        try:
            pick = np.nanargmin([next_gps_time, next_imu_time, next_las_time])
        except ValueError:
            break
        if pick == 0:
            # Generate gps_odom message
            h =  std_msgs.msg.Header()
            h.seq = count
            timestamp = unix_to_rostime(next_gps_time)
            h.stamp = unix_to_rostime(next_gps_time)
            h.frame_id = BASE_FRAME_ID
            t = TransformStamped()
            t.header = h
            t.child_frame_id = GPS_FRAME_ID
            t.transform.translation.x = next_gps_x_y_z[0]
            t.transform.translation.y = next_gps_x_y_z[1]
            t.transform.translation.z = next_gps_x_y_z[2]
            q = tf.transformations.quaternion_from_euler(0.,0.,next_gps_theta)
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            tfmsg = TFMessage()
            tfmsg.transforms.append(t)
            # Write message
            bag.write(TF_TOPIC, tfmsg, timestamp)
            # Load next index
            next_gps_indx = next_gps_indx + 1
            # Fetch next gps observation.
            next_gps_time, next_gps_x_y_z, next_gps_theta, next_gps_cov = fetch_gps(
                    gps_times,
                    gps_x_y_z,
                    gps_theta,
                    gps_cov,
            )
        if pick == 1:
            # Generate imu_odom message
            h =  std_msgs.msg.Header()
            h.seq = count
            timestamp = unix_to_rostime(next_imu_time)
            h.stamp = unix_to_rostime(next_imu_time)
            h.frame_id = BASE_FRAME_ID
            t = TransformStamped()
            t.header = h
            t.child_frame_id = IMU_FRAME_ID
            t.transform.translation.x = next_imu_pos[0]
            t.transform.translation.y = next_imu_pos[1]
            t.transform.translation.z = next_imu_pos[2]
            t.transform.rotation.x = next_imu_q[0]
            t.transform.rotation.y = next_imu_q[1]
            t.transform.rotation.z = next_imu_q[2]
            t.transform.rotation.w = next_imu_q[3]
            tfmsg = TFMessage()
            tfmsg.transforms.append(t)
            # Write message
            bag.write(TF_TOPIC, tfmsg, timestamp)
            # Load next index
            next_imu_indx = next_imu_indx + 1
            # Fetch next imu observation.
            next_imu_time, next_imu_pos, next_imu_q = fetch_imu(
                    imu_times,
                    imu_pos,
                    imu_q,
            )
        if pick == 2:
            # Generate velodyne message
            h =  std_msgs.msg.Header()
            h.seq = count
            timestamp = unix_to_rostime(next_las_time)
            h.stamp = unix_to_rostime(next_las_time)
            h.frame_id = LAS_FRAME_ID
            points = pc2.create_cloud_xyz32(h, next_las_XYZ)
            # Write message
            bag.write(LAS_TOPIC, points, timestamp)
            # Load next index
            next_las_indx = next_las_indx + 1
            # Fetch next laser observation
            next_las_time, next_las_XYZ = fetch_las(next_las_indx, las_dir, las_files)
        print("{} out of {}".format(count, total_messages))
finally:
    bag.close()


