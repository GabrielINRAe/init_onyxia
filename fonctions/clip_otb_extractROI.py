import os
import sys
import glob

def get_images_path(dir_images, ext = ".tif"):
    """
    Get the path of all the images in a directory.  

    Parameters:
        dir_images (str): Path to the directory containing the images.
        ext (str): Extension of the images, default is '.tif'.

    Returns:
        l_path (str) : List of the path of the images.
    """
    l_path = sorted(glob.glob(os.path.join(dir_images, "*" + ext)))  # Trier pour assurer l'ordre
    return l_path

def script_otb_extractROI(images_path, vectorfile, output_dir, dtype = 'int16'):
    """
    Clip une ou un esemble d'images avec un fichier vecteur.

    Parameters:
        images_path (list): Path to the images to clip.
        vectorfile (str): Path to the vector file.
        output_dir (str): Path to the output directory.
        dtype (str): Data type of the output images, defaut is 'int16'.

    Returns:
        cmd_f (str): Command to execute in the terminal.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd_pattern = ("otbcli_ExtractROI -in {in_image} -mode fit -mode.fit.vect {vectorfile}  -out {out_image} {dtype}")
    cmd_f = ""
    for i in images_path:
        name_image = os.path.basename(i)
        out_image = os.path.join(output_dir, name_image[:-4] + "_clip.tif")
        cmd = cmd_pattern.format(in_image=i, vectorfile=vectorfile, out_image=out_image, dtype=dtype)
        cmd_f += cmd + "\n"
    return cmd_f