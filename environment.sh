GIT_REPO=fosses_talus_detection
git clone --depth 1 https://gabriel.orabona:AHh14xclHvx7CyI-nZnHy286MQp1Ojcxawk.01.0z0n9h5ds@forge.inrae.fr/dynafor/dev-test/gabriel-orabona/${GIT_REPO}.git
chown -R onyxia:users ${GIT_REPO}/

pip install -r ${GIT_REPO}/requirements.txt

mc cp -r s3/gabrielorabona/diffusion/haies_mnt_lidar .
