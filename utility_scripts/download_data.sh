#!/bin/bash

if [[$1 = "small"]]
then
wget "https://seafile.cloud.uni-hannover.de/f/d1752d0ec90148ddb1bb/?dl=1" -O datasmall.zip
unzip datasmall.zip
rm datasmall.zip
fi

if [[$1 = "large"]]
then
wget "https://seafile.cloud.uni-hannover.de/f/94ac34a318f2449180df/?dl=1" -O datalarge.zip
unzip datalarge.zip
rm datalarge.zip
fi

if [[$1 = "models"]]
then
wget "https://seafile.cloud.uni-hannover.de/f/a7329732a61c4a7383a4/?dl=1" -O models.zip
unzip models.zip
rm models.zip
fi

if [[$1 = "results"]]
then
wget "https://seafile.cloud.uni-hannover.de/f/20fe4c6c2b874fd79106/?dl=1" -O results.zip
unzip results.zip
rm results.zip
fi

if [[$1 = "all"]]
then
wget "https://seafile.cloud.uni-hannover.de/f/d1752d0ec90148ddb1bb/?dl=1" -O datasmall.zip
unzip datasmall.zip
rm datasmall.zip

wget "https://seafile.cloud.uni-hannover.de/f/a7329732a61c4a7383a4/?dl=1" -O models.zip
unzip models.zip
rm models.zip

wget "https://seafile.cloud.uni-hannover.de/f/94ac34a318f2449180df/?dl=1" -O datalarge.zip
unzip datalarge.zip
rm datalarge.zip

wget "https://seafile.cloud.uni-hannover.de/f/20fe4c6c2b874fd79106/?dl=1" -O results.zip
unzip results.zip
rm results.zip
fi
