# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 14:12:00 2018
@author: yhamrouni
"""

import glob
import os
import sys
# to sort naturally a dict
import natsort
from collections import OrderedDict
import zipfile
import fnmatch

def get_metadata_from_image(in_image_path):
    """
    Get the metadata according to the Sentinel-2 image name.

    Parameters
    ----------
    in_image_path : str

    Returns
    -------
    metadata : str

    """
    plateform_date = os.path.basename(in_image_path)[0:19]
    tile = os.path.basename(in_image_path)[35:41]
    metadata = plateform_date + "_" + tile
    return (metadata)


# The zip folder
unzip = True
in_dir_raw ='/home/onyxia/work/L3/RAR'
in_dir = "/home/onyxia/work/L3/in_dir"
out_dir = "/home/onyxia/work/L3/out_dir"
in_dates_file = in_dir + '/in_dates.txt'


# unzip = sys.argv[1]
# in_dir = sys.argv[2]
# out_dir = sys.argv[3]
# #in_dir_raw = ''
# in_dates_file = in_dir + '/in_dates.txt'
# # unzip=False

# unzip = True
# in_dir_raw = sys.argv[4]


# Individual date stack
# for in_folder in glob.glob(in_dir + "/*SENTINEL2X_*"):
#     bands = []
#     images = []
#     for i in glob.glob(in_folder+"/*FRC*.tif"):
#         band = i.split('FRC_')[1].split('.')[0]
#         bands.append(band)
#         images.append(i)
#     dic = dict(zip(bands, images))
#     keys = natsort.natsorted(dic.keys())
#     dic_sorted = dict(OrderedDict((k, dic[k]) for k in keys))
#     list_tmp = list(dic_sorted.values())
#     string = ""
#     for k in range(len(list_tmp)):
#         string += list_tmp[k]+" "
#     out_stack = out_dir + "/" \
#         + str(get_metadata_from_image(in_folder)) \
#         + "_Stack_10m.vrt"
#     process = "gdalbuildvrt -tr 10 10 -separate "\
#         + out_stack + " "\
#         + str(string)
#     print(process)
#     os.system(process)

# SITS stack
dates = []
# Unzip Sentinel-2 level 3 images
if unzip == True:
    for image in glob.glob(in_dir_raw + "/*.zip"):
        print("unzip ", str(image))
        os.system(" unzip " + image + " *FRC* *_R1* -d " + in_dir)
        print("unzipping ", str(image))
        date = os.path.basename(image)[11:19]
        dates.append(date)
        # os.remove(image)
else:
    for image in glob.glob(in_dir + "/*SENTINEL2X_*"):
        date = os.path.basename(image)[11:19]
        dates.append(date)
dates.sort()
with open(in_dates_file, "w") as f:
    for i in dates:
        f.write(str(i) + "\n")
f.close()
print(dates)

string = ""
folder_list = []
for date in dates:
    for in_folder in glob.glob(in_dir+"/*/"):
        if fnmatch.fnmatch(in_folder, '*' + date + '*'):
            folder_list.append(in_folder)

for in_folder in folder_list:
    bands = []
    images = []
    for i in glob.glob(in_folder+"/*FRC*.tif"):
        band = i.split('FRC_')[1].split('.')[0]
        bands.append(band)
        images.append(i)
    dic = dict(zip(bands, images))
    keys = natsort.natsorted(dic.keys())
    dic_sorted = dict(OrderedDict((k, dic[k]) for k in keys))
    list_tmp = list(dic_sorted.values())
    for k in range(len(list_tmp)):
        string += list_tmp[k]+" "
    out_stack = out_dir + "/" \
        + 'SENTINEL2X_'\
        + dates[0][:4]\
        + "_Stack_10m.vrt"
    process = "gdalbuildvrt -tr 10 10 -separate "\
        + out_stack + " "\
        + str(string)
    print(process)
    os.system(process)


























