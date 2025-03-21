# Pour associer les images L3 en un netcdf

from fonctions.sentinel_processor import SentinelProcessor
import glob
from pathlib import Path
import natsort
from collections import OrderedDict
import os

in_dir = r"/home/onyxia/work/L3/in_dir"
path_list = sorted(glob.glob(pathname=os.path.join(in_dir,"**/*FRC*.tif"), recursive=True))

output_path = r"/home/onyxia/work/L3/out_dir/S2_L3_2020.nc"
bb_path = r"/home/onyxia/work/data/masque_etangs.tif"
custom_chunks = {"x": 100, "y": 100, "time": 1}
target_crs = "EPSG:2154"
processor = SentinelProcessor(
    file_list=path_list, 
    dir_global=in_dir, 
    output_path=output_path, 
    crs=target_crs,
    clip_ref=bb_path)
array = processor.build_dataset(chunking="auto")
processor.save_dataset()

# list_array = list(array.groupby("time.month"))
# for i in list_array:
#     array_date = i[1]
#     date = str(array_date.time.values[0])[:7]
#     array_date.to_netcdf(os.path.join(r"E:\Stage_PNRBrenne\Data\Data_Sentinel\L3\S2_concat\emprise_reduite",f"{date}.nc"))