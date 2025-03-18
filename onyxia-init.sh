GIT_REPO=stage_pnr_dynafor
git --depth clone https://forgemia.inra.fr/dynafor/dev-test/gabriel-orabona/${GIT_REPO}.git
chown -R onyxia:users ${GIT_REPO}/

conda create env -f environment.yaml
conda activate stage

# conda install -y xarray
# conda install -y rioxarray
# conda install -y geopandas
# conda install -y gdal==3.10.1
# conda install -y holoviews
# conda install -y hvplot
# conda install -y sklearn
