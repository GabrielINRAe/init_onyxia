git clone https://oauth2:z9nVf3NenWpJyGxfrnD9@forgemia.inra.fr/dynafor/dev-test/gabriel-orabona/stage_pnr_dynafor.git
chown -R onyxia:users stage_pnr_dynafor/
git clone https://oauth2:z9nVf3NenWpJyGxfrnD9@forgemia.inra.fr/dynafor/dev-test/lucas/spectral-indices.git
chown -R onyxia:users spectral-indices/


# conda install -y xarray
# conda install -y rioxarray
# conda install -y geopandas
# conda install -y gdal==3.10.1
conda install -y holoviews
conda install -y hvplot
# conda install -y sklearn
# conda install -y natsort
conda install -y jupyter_bokeh
# conda install -y libgdal-netcdf
# conda install -y cftime
# conda install -y scikit-image

mc cp -r s3/gabgab/diffusion/data /home/onyxia/work
