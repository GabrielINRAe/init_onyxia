import xarray as xr
import numpy as np
import pandas as pd
import rioxarray as rio
import glob
import os
import re
from pathlib import Path
import geopandas as gpd

def extract_date_band(filepath):
    """Extrait la date (YYYYMMDD) et la bande (B2, B3, ..., B8A, B11, etc.) du nom du fichier."""
    match_date = re.search(r"(\d{8})", Path(filepath).stem)
    match_band = re.search(r"(B\d{1,2}A?)", Path(filepath).stem)
    
    if match_date and match_band:
        date = pd.to_datetime(match_date.group(1), format="%Y%m%d")
        band = match_band.group(1)
        return date, band
    else:
        raise ValueError(f"Impossible d'extraire date/bande depuis : {filepath}")

class SentinelProcessor:
    def __init__(self, file_list, dir_global, output_path, crs:str, bbox=None, clip_ref=None):
        """
        Initialise le processeur Sentinel.
        
        :param file_list: Liste des chemins vers les fichiers TIF à traiter.
        :dir_global: Path vers le dossier où les images/masques sont stockées
        :param output_path: Chemin de sortie du fichier NetCDF.
        :param crs: crs utilisé pour les traitements
        :param bbox: Bounding box (xmin, ymin, xmax, ymax) ou None si pas de clipping.
        :param clip_ref: Chemin vers un raster ou shapefile servant de référence pour le clipping. 
                         Si None, on utilise bbox si fourni.
        """
        self.dir_global = Path(dir_global)
        self.file_list = [Path(f) for f in file_list]
        self.output_path = Path(output_path)
        self.bbox = bbox
        self.clip_ref = Path(clip_ref) if clip_ref else None
        self.dataset = None
        self.crs = crs

    def clip_data(self, da):
        """Applique le clipping à un raster en fonction du raster/shapefile ou de la bbox fournie."""
        if self.clip_ref:
            if self.clip_ref.suffix in [".shp", ".geojson"]:
                print(f"✂️ Clipping avec le shapefile : {self.clip_ref}")
                shape = gpd.read_file(self.clip_ref)
                da = da.rio.clip(shape.geometry, shape.crs, drop=True)
            elif self.clip_ref.suffix in [".tif", ".tiff"]:
                print(f"✂️ Clipping avec le raster : {self.clip_ref}")
                ref_raster = rio.open_rasterio(self.clip_ref)
                da = da.rio.clip_box(*ref_raster.rio.bounds(), crs=ref_raster.rio.crs)
        elif self.bbox:
            print(f"✂️ Clipping avec la bounding box manuelle : {self.bbox}")
            da = da.rio.clip_box(*self.bbox, crs=self.crs)  # Gestion automatique du CRS
        return da
    
    def build_dataset(self, chunking=None):
        """Construit un xarray.Dataset à partir des fichiers chargés."""
        data_dict = {}
        dates = set()
                
        for file in self.file_list:
            date, band = extract_date_band(file)
            dates.add(date)
            print(f"📂 Chargement : {file.name} (Bande {band}, Date {date.date()})")
            
            # Charger l'image en float32
            da = rio.open_rasterio(file).astype(np.float32)
            da = da.squeeze("band", drop=True)  # Suppression de la dimension inutile "band"

            # Rééchantillonnage à 10m si nécessaire
            if da.rio.resolution()[0] != -10:
                print(f"🔄 Rééchantillonnage de {band} à 10m...")
                da = da.rio.reproject_match(self.get_reference_raster())

            # TODO Ajouter le masque des nuages
            updir = os.path.dirname(Path(file))
            m_path = glob.glob(pathname=os.path.join(updir, "**/*FLG*"), recursive=True)[0]
            print(f"Application du masque {os.path.basename(m_path)}")
            masque = xr.open_dataarray(m_path)            
            da = xr.where(masque!=1,da,np.nan).rio.write_crs(self.get_reference_raster().rio.crs)
            
            # Reprojection
            print(f"🔄 Reprojection du raster vers {self.crs} avant clipping...")
            da = da.rio.reproject(self.crs)
            da = da.rio.write_crs(self.crs)

            # Clipping APRES rééchantillonnage
            if self.clip_ref:
                ref_raster = rio.open_rasterio(self.clip_ref)
                da = xr.align(ref_raster,da, join="left",exclude="time")[1]
            # if self.clip_ref or self.bbox:
            #     da = self.clip_data(da)
            
            # Stockage des données dans le dictionnaire
            if band not in data_dict:
                data_dict[band] = []
            data_dict[band].append((date, da))
        
        # Trier et concaténer les données par bande
        dataset = {}
        for band, values in data_dict.items():
            values.sort(key=lambda x: x[0])  # Trier par date
            dataset[band] = xr.concat([v[1] for v in values], dim=pd.Index([v[0] for v in values], name="time"))
        
        self.dataset = xr.Dataset(dataset)
        if self.dataset["band"]:
            self.dataset = self.dataset.squeeze("band", drop=True)
        self.dataset = self.dataset.rio.write_crs(self.crs)
        if chunking:
            self.dataset = self.dataset.chunk(chunking)            
        return(self.dataset)

    def get_reference_raster(self):
        """
        Sélectionne une image de référence pour le rééchantillonnage (une bande à 10m).
        """
        for file in self.file_list:
            _, band = extract_date_band(file)
            if band in ["B2", "B3", "B4", "B8"]:  # Bandes déjà à 10m
                print(f"🎯 Image de référence : {file.name}")
                return xr.open_dataarray(file)
        raise ValueError("Aucune image de référence à 10m trouvée pour le rééchantillonnage.")
    
    def save_dataset(self):
        """Enregistre le dataset en NetCDF."""
        if self.dataset is None:
            raise ValueError("Le dataset n'a pas été construit. Appelez build_dataset() avant save_dataset().")
        self.dataset.to_netcdf(self.output_path)
        print(f"✅ Dataset enregistré à : {self.output_path}")
