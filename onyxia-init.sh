GIT_REPO=stage_pnr_dynafor
git --depth clone https://forgemia.inra.fr/dynafor/dev-test/gabriel-orabona/${GIT_REPO}.git
chown -R onyxia:users ${GIT_REPO}/


conda install -y xarray
conda install -y rioxarray
conda install -y geopandas
conda install -y gdal==3.10.1
conda install -y holoviews
conda install -y hvplot
conda install -y sklearn
conda install -y natsort
conda install -y jupyter_bokeh

mkdir -p L3/{in_dir,out_dir} prod

mc cp -r s3/gabgab/diffusion/data /home/onyxia/work
mc cp -r s3/gabgab/diffusion/RAR /home/onyxia/work/L3
