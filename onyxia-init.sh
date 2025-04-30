GIT_REPO=stage_pnr_dynafor
git --depth clone https://oauth2:z9nVf3NenWpJyGxfrnD9@forgemia.inra.fr/dynafor/dev-test/gabriel-orabona/${GIT_REPO}.git
chown -R onyxia:users ${GIT_REPO}/
GIT_REPO2=spectral-indices
git --depth clone https://oauth2:z9nVf3NenWpJyGxfrnD9@forgemia.inra.fr/dynafor/dev-test/lucas/${GIT_REPO2}.git
chown -R onyxia:users ${GIT_REPO2}/


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
