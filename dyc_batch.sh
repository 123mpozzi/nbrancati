#!/bin/bash

# variables
x_path="dataset/ecu/original_images"
x_ext="jpeg"

pred_path="pred"
pred_ext="png"

# make dirs
mkdir -p $x_path
mkdir -p $pred_path

for x in "${x_path}/*.${x_ext}"
do
  pred="${pred_path}/${x}.${pred_ext}"

  # run the skin detector
  # dyc x_in pred_out
  docker-compose exec opencv ./app $x $pred
  #docker-compose exec opencv ./app image1.jpg image2.jpg
done



### NUOVO METODO:
# fai partire il venv datloader nella cartella workspace
# (fai partire docker con up)
# (muovi i dataset nella cartella dyc/dataset se non ci sono)
# poi vai in cartella dyc e fai partire dyc_batch_py.py
###




## new

# faccio partire docker bash
docker-compose up -d opencv
docker-compose exec opencv bash

#faccio cd nella cartella dove devono arrivare gli output

# faccio partire il loop dei dataset
# for img in dataset/ecu/original_images/*; do ./app $img ./pred/$img; done
for img in ../dataset/ecu/original_images/*.jpeg; do echo $img;../app $img `basename $img | sed s/\.jpeg/_out_.png/`; done
