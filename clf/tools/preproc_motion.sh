DATAPATH=$1
# generate motion map
echo Extracting motion map ...
python scripts/data/motion_map.py $DATAPATH  && echo DONE || echo Run FAILED, please check
