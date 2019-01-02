"""
Open Images Dataset V4 database Maker

```sh
$ python make_database_oidv4.py /path/to/train_annotations.csv /path/to/oidv4
```
"""
from dlcliche.utils import *
from od_anno import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('csv_file')
parser.add_argument('OIDV4_folder')
args = parser.parse_args()

print(f'Making database for Open Images Dataset V4 as {args.csv_file}. * will be 1.8G bytes long.')
print(f'Reading annotations from {args.OIDV4_folder}...')
ODAnno.from_google_open_images_v4(args.csv_file, args.OIDV4_folder)
