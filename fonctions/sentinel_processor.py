import xarray as xr
import numpy as np
import pandas as pd
import rioxarray as rio
import glob
import os
import re
from pathlib import Path

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
    def __init__(self, file_list, dir_global, output_path, crs: str, ref_raster_path: str):
        """
        Initialise le processeur Sentinel-2. Attention pour l'instant le code ne 
        fonctionne qu'avec un raster de référence ayant déjà la bonne résolution, la bonne emprise et la bonne projection.

        Paramètres :
        ------------
        file_list : list[str]
            Liste des chemins des fichiers TIF contenant les bandes Sentinel-2 à traiter.
        dir_global : str
            Chemin vers le répertoire global contenant les images Sentinel-2.
        output_path : str
            Chemin où sera sauvegardé le fichier NetCDF final contenant la série temporelle.
        crs : str
            Code EPSG du système de coordonnées souhaité (ex. "EPSG:2154").
        ref_raster_path : str
            Chemin vers le raster de référence servant à la reprojection, au rééchantillonnage et au clip.

        Attributs :
        ----------
        self.dir_global : Path
            Répertoire contenant les données Sentinel-2.
        self.file_list : list[Path]
            Liste des fichiers images convertis en objets `Path`.
        self.output_path : Path
            Chemin de sortie sous forme d'un objet `Path`.
        self.ref_raster_path : Path
            Chemin du raster de référence sous forme d'un objet `Path`.
        self.dataset : xarray.Dataset ou None
            Contiendra le dataset final une fois construit.
        self.crs : str
            Système de coordonnées de sortie.
        self.ref_raster : xarray.DataArray
            Raster de référence chargé pour la reprojection et le rééchantillonnage. 
        """

        # TODO Ajouter des options de clip à partir de shp ou d'une bounding box

        self.dir_global = Path(dir_global)
        self.file_list = [Path(f) for f in file_list]
        self.output_path = Path(output_path)
        self.ref_raster_path = Path(ref_raster_path)
        self.dataset = None
        self.crs = crs
        self.ref_raster = self.load_ref_raster()

    def load_ref_raster(self):
        """Charge le raster de référence et vérifie qu'il possède bien un CRS défini."""
        ref_raster = rio.open_rasterio(self.ref_raster_path).squeeze()
        if ref_raster.rio.crs is None:
            raise ValueError(f"Le raster de référence {self.ref_raster_path} n'a pas de CRS défini.")
        return ref_raster

    def build_dataset(self, chunking=None):
        """Construit un xarray.Dataset à partir des fichiers chargés."""
        data_dict = {}
        dates = set()
                
        for file in self.file_list:
            date, band = extract_date_band(file)
            dates.add(date)
            print(f"📂 Chargement : {file.name} (Bande {band}, Date {date.date()})")
            
            # Charger l'image en float32
            da = rio.open_rasterio(file).astype(np.float32).squeeze("band", drop=True)

            # Reprojection et alignement au raster de référence
            if da.rio.crs != self.ref_raster.rio.crs:
                print(f"🔄 Reprojection de {band} vers {self.ref_raster.rio.crs}...")
                da = da.rio.reproject(self.ref_raster.rio.crs)

            print(f"🔄 Rééchantillonnage et alignement de {band}...")
            da = da.rio.reproject_match(self.ref_raster)

            # Charger et aligner le masque de nuages
            updir = os.path.dirname(Path(file))
            m_path = glob.glob(pathname=os.path.join(updir, "**/*FLG*"), recursive=True)[0]
            print(f"📂 Chargement du masque : {os.path.basename(m_path)}")
            masque = rio.open_rasterio(m_path).squeeze("band", drop=True)

            if masque.rio.crs != self.ref_raster.rio.crs:
                print(f"🔄 Reprojection du masque de nuages...")
                masque = masque.rio.reproject(self.ref_raster.rio.crs)

            print(f"🔄 Alignement du masque de nuages...")
            masque = masque.rio.reproject_match(self.ref_raster)

            # Appliquer le masque de nuages
            da = xr.where(masque != 1, da, np.nan).rio.write_crs(self.ref_raster.rio.crs)

            # Stockage des données dans le dictionnaire
            if band not in data_dict:
                data_dict[band] = []
            data_dict[band].append((date, da))
        
        # Trier et concaténer les données par bande
        dataset = {}
        for band, values in data_dict.items():
            values.sort(key=lambda x: x[0])  # Trier par date
            dataset[band] = xr.concat(
                [v[1] for v in values], 
                dim=pd.Index([v[0] for v in values], name="time")
            )
        
        self.dataset = xr.Dataset(dataset).rio.write_crs(self.ref_raster.rio.crs)
        if chunking:
            self.dataset = self.dataset.chunk(chunking)
        return self.dataset

    def save_dataset(self):
        """Enregistre le dataset en NetCDF."""
        if self.dataset is None:
            raise ValueError("Le dataset n'a pas été construit. Appelez build_dataset() avant save_dataset().")
        self.dataset.to_netcdf(self.output_path)
        print(f"✅ Dataset enregistré à : {self.output_path}")
