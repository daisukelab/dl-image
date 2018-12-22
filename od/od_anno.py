from dlcliche.utils import *
from dlcliche.image import *
from PIL import Image
import collections

logger = get_logger()

class Dataset(object):
    """Similar definition with pytorch."""

    def __init__(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        return len(self.X)

class ODAnnoDataset(Dataset):
    def __init__(self, X, y, classes, path=''):
        super().__init__(X, y)
        self.classes, self.path = classes, Path(path)

    def __getitem__(self, index):
        filename, y = super().__getitem__(index)
        return load_rgb_image(self.path/filename), y

    def show(self, index_or_iter, matrixsize=None, figsize=(12, 12)):
        if not isinstance(index_or_iter, collections.Iterable):
            index_or_iter = [index_or_iter]
        if not matrixsize: matrixsize = [1, len(index_or_iter)]
        axes = subplot_matrix(rows=matrixsize[1], columns=matrixsize[0],
                              figsize=figsize)
        for idx, ax in zip(index_or_iter, axes):
            img, y = self[idx]
            for _y in y:
                show_image(img, ax=ax)
                ax_draw_bbox(ax, self.bbox(_y), self.classes[self.label(_y)])
        plt.show()

    @staticmethod
    def bbox(y): return y[:4]
    @staticmethod
    def label(y): return np.argmax(y[4:]) if 1 < len(y[4:]) else int(y[4:])

class ODAnno(object):
    """Annotation dataset.
    
    # Basic design
    - Has a data frame object to hold annotations.
    - Has a csv filename to hold filename for annotation file.
    
    # Cautions
    - Filenames stored in anno_df.File could have simple filename or relative path.
    - Subset have a single image folder, whatever split images belongs to. 
    """

    def __init__(self, anno_csv, image_folder, anno_df=None):
        """ODAnno constructor.

        Arguments:
            anno_csv: Path name for 'annotations.csv', or None if you use dataframe only.
            image_folder: Folder path name where images are stored.
            anno_df: DataFrame that have ODAnno info in advance.
        """
        self.anno_csv, self.image_folder = anno_csv, Path(image_folder)
        self.anno_df = pd.read_csv(anno_csv) if anno_df is None else anno_df
        self._reset_classes()

    def __len__(self):
        return len(self.anno_df)

    def _reset_classes(self):
        self.classes = sorted(list(set(self.anno_df.Label)))

    def save(self):
        self.anno_df.to_csv(self.anno_csv, index=False)

    def save_as(self, filename):
        self.anno_csv = filename
        self.save()

    def _set_size(self):
        """Set column 'Size' if not there."""
        if 'Size' in self.anno_df.columns: return
        w_label = (self.anno_df.XMax - self.anno_df.XMin).values
        h_label = (self.anno_df.YMax - self.anno_df.YMin).values
        self.anno_df['Size'] = w_label * h_label

    def filter_by_label_(self, labels):
        """Extract classes you need, done in place."""
        self.anno_df = self.anno_df[self.anno_df.Label.isin(labels)]
        self._reset_classes()
        return self

    def filter_by_size_(self, thresh_min=None, thresh_max=None):
        """Filter by label size (volume).
        Size is simply width x hight. Both width and height are in range [0, 1].
        Filter all labels other than its size is in range [thresh_min <= size < thresh_max].

        Arguments:
            thresh_min: If set, pass through labels: `thresh_min <= size`.
            thresh_max: If set, pass through labels: `size < thresh_max`.
        """
        self._set_size()
        if thresh_min is not None:
            self.anno_df = self.anno_df[self.anno_df.Size >= thresh_min]
        if thresh_max is not None:
            self.anno_df = self.anno_df[self.anno_df.Size < thresh_max]
        self._reset_classes()
        return self
   
    def filter_largest_(self):
        """Filter small labels except biggest one from anno_df.
        This is useful for making classification dataset.
        """
        self._set_size()
        # Filter biggest
        idx = self.anno_df.groupby('ImageID')['Size'].transform(max) == self.anno_df['Size']
        self.anno_df = self.anno_df[idx]
        self._reset_classes()
        return self

    def __files(self, df=None):
        if df is None: df = self.anno_df
        return [str(self.image_folder/f) for f in df.File]

    def _coco_df(self):
        """Convert to MS COCO format."""
        df, outdf = self.anno_df, pd.DataFrame()
        outdf['File'] = self.__files(df)
        outdf['Label'] = df.Label
        outdf['w'] = (df.XMax - df.XMin) * df.Width
        outdf['h'] = (df.YMax - df.YMin) * df.Height
        outdf['x'] = df.XMin * df.Width
        outdf['y'] = df.YMin * df.Height
        return outdf

    def dataset(self, one_hot=False):
        def or_one_hot(cls):
            return np.eye(len(self.classes))[cls] if one_hot else cls
        groups = self._coco_df().groupby('File')
        X, y = [], []
        for filename, df in groups:
            _y = np.array([np.r_[r[['x', 'y', 'w', 'h']].values,
                                 or_one_hot(self.classes.index(r.Label))]
                                 for _, r in df.iterrows()])
            X.append(filename)
            y.append(_y)
        return ODAnnoDataset(X, y, classes=self.classes, path='')


    @staticmethod
    def from_google_open_image_v4(anno_csv, open_image_folder):
        """Convert OpenImage V4 dataset annotation, save it,
        then generate annotation instance.

        CAUTION: Takes 2 hours or so.
        """

        OIDPATH = Path(open_image_folder)
        annos = df_load_excel_like(OIDPATH/'train-annotations-bbox.csv')
        class_descs = pd.read_csv(OIDPATH/'class-descriptions-boxable.csv', header=None)
        class_descs.columns = ['id', 'name']
        # Convert label id to name
        label_id2name = pd.Series(class_descs.name.values, index=class_descs.id).to_dict()
        annos['Label'] = annos['LabelName'].apply(lambda x:label_id2name[x])
        annos['Split'] = 'train'
        # Make filenames
        annos['File'] = annos.ImageID.apply(lambda x: str(Path('train/'+x).with_suffix('.jpg')))
        # Get image size
        sizes = np.array([Image.open(OIDPATH/filename).size for filename in annos.File])
        annos['Width'] = sizes[:, 0]
        annos['Height'] = sizes[:, 1]
        # Make final dataframe & return it
        od_anno_df = annos[['ImageID', 'File', 'Label', 'XMin', 'XMax', 'YMin', 'YMax', 'Width', 'Height', 'Split']]
        # Make instance and save.
        anno = ODAnno(anno_csv, open_image_folder, anno_df=od_anno_df)
        anno.save()
        return anno

    def filter_google_open_image_v4_confident_only_(self, open_image_folder):
        df = df_load_excel_like(Path(open_image_folder)/'train-annotations-human-imagelabels-boxable.csv')
        confident_imageids = df[df.Confidence != '0'].index
        self.anno_df = self.anno_df[self.anno_df.index.isin(confident_imageids)]

    def brew_subset(self, new_anno_csv, dest_folder=None, resize_shape=None):
        # Make list of files, then check parent folder consistency.
        dest_folder = Path(dest_folder)
        files = self.__files()
        # Resize and make all copies.
        if dest_folder is not None:
            if resize_image_files(dest_folder=dest_folder,
                                  source_files=list(set(files)),
                                  shape=resize_shape,
                                  skip_if_any_there=True) is None:
                logger.info('Subset brew skipped resize operation, looks done already.')
        # Set resized shape
        self.anno_df.Width  = resize_shape[0]
        self.anno_df.Height = resize_shape[1]
        # Rebase folder
        if dest_folder is not None:
            self.image_folder = dest_folder
            self.anno_df.File = [str(Path(file).name) for file in list(self.anno_df.File)]
        self.save_as(new_anno_csv)

    def build_score_df_sorted(self, label_weight):
        """Build score of annotation per image as a dataframe.
        This score is for selecting samples for your porpose.
        """
        # Prepare data frames
        boxes = self.anno_df[self.anno_df.Label.isin(label_weight.keys())]
        label_score = pd.DataFrame(boxes[['ImageID', 'Label']]).sort_values(['ImageID'])
        score = pd.DataFrame(sorted(boxes.ImageID.unique()), columns=['ImageID'])
        for key in label_weight:
            label_score[key] = 0.0 # set all 0, then set label weight as its score 
            label_score[key][label_score.Label == key] = label_weight[key]
        labelscore_group = label_score.groupby('ImageID')
        # Assign label's score for each image
        # One image could have multiple labels, then set sum of them
        sum_score = np.zeros((len(score), len(label_weight)))
        for i in range(len(score)):
            iid = score.ImageID[i]
            sum_values = labelscore_group.get_group(iid).sum(axis=0)
            for j, key in enumerate(label_weight):
                sum_score[i, j] = sum_values[key]
        for j, key in enumerate(label_weight):
            score[key] = sum_score[:, j]
        # Finally, calculate total score for images
        score['raw_score'] = score[list(label_weight.keys())].sum(axis=1)
        # And also calculate normalized score
        score_max = np.max(sum_score, axis=-1)
        norm_score = (sum_score.T / score_max).T
        score['normalized_score'] = np.sum(norm_score, axis=-1)
        # Return sorted one
        sorted_score = score.sort_values(['normalized_score'], ascending=False)
        # With extra column for counting.
        sorted_score['count'] = 1
        return sorted_score
