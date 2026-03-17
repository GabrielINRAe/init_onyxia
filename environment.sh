GIT_TOKEN=AHh14xclHvx7CyI-nZnHy286MQp1Ojcxawk.01.0z0n9h5ds
GIT_REPO=fosses_talus_detection

git clone https://gabriel.orabona:${GIT_TOKEN}@forge.inrae.fr/dynafor/dev-test/gabriel-orabona/${GIT_REPO}.git


git -C ${GIT_REPO} remote set-branches origin '*'
git -C ${GIT_REPO} fetch --all

chown -R onyxia:users ${GIT_REPO}/

pip install -r ${GIT_REPO}/requirements.txt
