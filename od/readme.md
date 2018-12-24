# Object Detector

This folder provides utilities for object detector (OD) application.

- Dataset handling.
- Example of training major OD solutions and (TBD) runtime code.

## Supported datasets

- Open Images Dataset V4.
- (TBD) MS COCO.
- (TBD) Pascal VOC.

## Dataset handling

Basic steps are:

1. Make annotation database once, which contains all the annotation data needed for further use.
2. Narrow down annotation labels for your application.
3. Make new subset dataset from original dataset based on selected labels.

Then the small subset of the big original dataset will be ready for your machine learning application training.

## Quick start

1. Run following to create database.
```sh
$ python make_database_oidv4.py /path/to/train_annotations.csv /path/to/oidv4
```
2. Try example notebook.
