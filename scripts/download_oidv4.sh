#!/bin/bash

echo Downloading Open Images Dataset V4 annotations...
wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-bbox.csv
wget https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-bbox.csv
wget https://storage.googleapis.com/openimages/2018_04/test/test-annotations-bbox.csv
wget https://storage.googleapis.com/openimages/2018_04/train/train-annotations-human-imagelabels-boxable.csv
wget https://storage.googleapis.com/openimages/2018_04/validation/validation-annotations-human-imagelabels-boxable.csv
wget https://storage.googleapis.com/openimages/2018_04/test/test-annotations-human-imagelabels-boxable.csv
wget https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv
wget https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv
wget https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv
wget https://storage.googleapis.com/openimages/2018_04/class-descriptions-boxable.csv
wget https://storage.googleapis.com/openimages/2018_04/train/train-image-ids-with-human-parts-and-mammal-boxes.txt
wget https://storage.googleapis.com/openimages/2018_04/class-ids-human-body-parts-and-mammal.txt
echo Done.

echo Downloading images ** This is not fully confirmed. We appreciate if you find any issues. **
wget https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/test.zip
wget https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/train_00.zip
wget https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/train_01.zip
wget https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/train_02.zip
wget https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/train_03.zip
wget https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/train_04.zip
wget https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/train_05.zip
wget https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/train_06.zip
wget https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/train_07.zip
wget https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/train_08.zip
wget https://datasets.figure-eight.com/figure_eight_datasets/open-images/zip_files_copy/validation.zip
echo Finished.
