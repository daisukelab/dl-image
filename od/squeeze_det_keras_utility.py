from dlcliche.utils import *
from dlcliche.image import *
import cv2
sys.path.append(str(Path(__file__).parent.parent/'externals'))
from kmeans_anchor_boxes.kmeans import (kmeans, avg_iou)

def optimize_anchors(ds, num_anchors):
    """Get optimized anchors for maximizing IoU on the dataset samples.
    
    # Examples
        ex) optimize_anchors(my_dataset, num_anchors=3)
        [[51.80000000000001, 27.73713600000002],
         [38.64, 45.63988800000001],
         [74.75999999999999, 73.5]]
    """
    y = np.array(flatten_list(ds.y))
    out = kmeans(y[:, 2:4], num_anchors)
    percentage = avg_iou(y[:, 2:4], out)
    print('IoU = ', percentage)
    print(str(out).replace(']\n ', '],\n').replace(' ', ', ').replace('\n[', '\n ['))

def make_training_files(dest_folder, odds, test_size):
    """SqueezeDetKeras label maker."""
    def _squeeze_det_keras_norm_classes(ds):
        """Normalize class names so that SqueezeDet system can handle."""
        normed = [c.lower().replace(' ', '_') for c in ds.classes]
        ds.classes = normed

    def _squeeze_det_keras_write_label(ds, gt_filename, label_folder):
        #'Human_face 0 0 0 100.800512 0.0 286.39948799999996 241.599744 0 0 0 0 0 0 0'
        label_folder = Path(label_folder)
        ensure_folder(label_folder)
        filenames = []
        for idx in range(len(ds)):
            fn = ds.X[idx]
            filename = str(label_folder/(Path(fn).stem+'.txt'))
            annos = [f'{ds.classes[ds.label(_y)]} 0 0 0 {ds.bx(_y)} {ds.by(_y)} {ds.bx(_y)+ds.bw(_y)} {ds.by(_y)+ds.bh(_y)} 0 0 0 0 0 0 0'
                     for _y in ds.y[idx]]
            with open(filename, 'w') as f:
                f.write('\n'.join(annos)+'\n')
            filenames.append(filename)        
        with open(gt_filename, 'w') as f:
            f.write('\n'.join(filenames)+'\n')

    def _squeeze_det_keras_make_one(dest_folder, target, ds):
        # Folders
        dest_folder = Path(dest_folder)
        # Write label and gt_{target}.txt
        _squeeze_det_keras_write_label(ds, dest_folder/f'gt_{target}.txt',
                                       dest_folder/'labels')
        # Write img_{target}.txt
        with open(dest_folder/f'img_{target}.txt', 'w') as f:
            f.write('\n'.join(ds.X)+'\n')

    # 
    _squeeze_det_keras_norm_classes(odds)
    train_ds, valid_ds = odds.train_test_split(test_size=test_size)
    _squeeze_det_keras_make_one(dest_folder, 'train', train_ds)
    _squeeze_det_keras_make_one(dest_folder, 'val', valid_ds)
    # config helper
    print('Copy followings to squeeze.config.')
    print('"CLASS_NAMES": '+str(odds.classes).replace("'", '"')+',')
    print('"CLASS_TO_IDX": '+str({c:i for i, c in enumerate(odds.classes)}).replace("'", '"')+',')

def calc_config_shape(shape):
    assert np.sum([s%16 for s in shape]) == 0
    print(f'"ANCHORS_HEIGHT": {int(shape[1]/16)},')
    print(f'"ANCHORS_WIDTH": {int(shape[0]/16)},')
    print(f'"IMAGE_HEIGHT": {int(shape[1])},')
    print(f'"IMAGE_WIDTH": {int(shape[0])},')


def load_gt_file(gt_file):
    with open(gt_file) as f:
        gts = f.read().splitlines()
    ids, labels, bboxes = [], [], []
    for filename in gts:
        _id = Path(filename).stem
        with open(filename) as f:
            _lines = [l.split(' ') for l in f.read().splitlines()]
        for a in _lines:
            ids.append(_id)
            labels.append(a[0])
            bboxes.append(np.array([float(a[4]), float(a[5]),
                          float(a[6])-float(a[4]), float(a[7])-float(a[5])]))
    gt_df = pd.DataFrame({'id':ids,'label':labels,'bbox':bboxes})
    return gt_df[['id','label','bbox']]

def show_gt_sample(image_folder, gt_df):
    groups = gt_df.groupby('id').groups
    for _id in groups:
        filename = str(Path(image_folder)/(_id+'.jpg'))
        img = load_rgb_image(filename)
        ax = show_image(img)
        for idx in groups[_id]:
            ax_draw_bbox(ax, gt_df.loc[idx]['bbox'], gt_df.loc[idx, 'label'])
        plt.figure(figsize=(12, 12))
        plt.show()
