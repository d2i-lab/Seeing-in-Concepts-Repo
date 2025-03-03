from enum import Enum
from typing import Optional
import base64
import json
import math
import io
from dataclasses import dataclass

from scipy.sparse import load_npz
import pandas as pd
import h5py
from pydantic import BaseModel
import numpy as np

from collections import Counter
import api.utils.settings as settings

settings = settings.Settings()

# from . import data_plugins

def get_concept_per_image(remap):
    '''
    Given remap of img -> {segment id -> concept id}, return
    img -> concept id -> [segment id]
    '''
    new_remap = {}
    for img in remap:
        new_remap[img] = {}
        for seg in remap[img]:
            concept = remap[img][seg]
            if concept not in new_remap[img]:
                new_remap[img][concept] = []
            new_remap[img][concept].append(seg)

    i1 = list(remap.keys())[0]

    return new_remap


def parse_bbox_str(bbox_str):
    return np.fromstring(bbox_str.decode('utf-8')[1:-1], dtype=float, sep=',')

class ExtensionTypes(str, Enum):
    PICKLE = 'pickle'
    HDF5 = 'hdf5'
    HDF5_EVAL = 'hdf5_eval'
    CSV = 'csv' # FOR SINGLE CSV EVALUATION ONLY

    @staticmethod
    def get_supported():
        return {v.value:v for v in ExtensionTypes}


# PLUGINS = data_plugins.PLUGINS


def discretize_col_by_cut(df, col, bins, qcut=False, fix_range=True):
    df_col_nonzero = df[col][df[col]> 1].copy()
    # df_col_nonzero = df[col][df[col]> 0]
    cut = None

    if len(df_col_nonzero) == 0:
        return

    if qcut:
        # print('Procesing colk', col)
        cut = pd.qcut(df_col_nonzero, bins, duplicates='drop')
    else:
        cut = pd.cut(df_col_nonzero, bins, duplicates='drop')

    if len(cut) == 0:
        return

    if fix_range:
        remap = {}
        for cat in list(cut.cat.categories):
            left = int(math.ceil(float(cat.left)))
            right = int(math.floor(float(cat.right)))

            lbracket = '('
            rbracket = ')'
            if left == right:
                remap[cat] = '{}'.format(left)
            if cat.closed == 'right':
                lbracket = '('
                rbracket = ']'
            elif cat.closed == 'left':
                lbracket = '['
                rbracket = ')'
            elif cat.closed == 'both':
                lbracket = '['
                rbracket = ']'

            if cat.left < left:
                lbracket = '['
            
            if cat.right > right:
                rbracket = ']'

            # if left == right:
            remap[cat] = '{}{}-{}{}'.format(lbracket, left, right, rbracket)
            # else:
                # remap[cat] = str(left)


        if len(set(remap.values())) != len(remap.values()):
            print('WARNING: Duplicate categories found.')
            return

        cut = cut.cat.rename_categories(remap)


    # Check if col is sparse
    if pd.api.types.is_sparse(df[col]):
        df[col] = df[col].sparse.to_dense()

    # df[col].iloc[cut.index] = cut
    df_col = df[col].copy()
    df_col[cut.index] = cut
    df[col] = df_col
    # df.loc[cut.index, col] = cut
    # df[col] = cut

def discretize_by_cut(df, bins, qcut=False):
    for col in df.columns:
        discretize_col_by_cut(df, col, bins, qcut=qcut)

def q_cut_percent(x, cuts):
    output = pd.qcut(x, cuts, duplicates='drop')
    # Create new_labels such that they indicate which percentile the category is in
    new_labels = []
    cuts_made = len(output.cat.categories)
    for i in range(cuts_made):
        new_labels.append('{}-{}%'.format(i*100//cuts, (i+1)*100//cuts))

    return output.cat.rename_categories(new_labels)


def get_concept_columns(hdf5_path):
    with h5py.File(hdf5_path, 'r') as f:
        return [l.decode('utf-8') for l in list(f['labels'])]

@dataclass
class LoadedDataset:
    df: pd.DataFrame
    file_name: str
    limit: int
    concept_mapping: dict
    label_mapping: dict
    n_concepts: int

    count_concepts: bool = False
    crowding: bool = False
    bbox_area: bool = False
    qcut: bool = False
    cut: bool = False
    dup_count: int = 0
    cid_labels: list = None
    merged_cids: dict = None

class Dataset(BaseModel):
    input_path: str
    input_ext: ExtensionTypes
    sparse_mat: Optional[object] = None
    n_concepts: Optional[int] = None

    count_concepts: Optional[bool] = True

    def hdf5_evaluation_iterator(self, limit: Optional[int] = None, **kwargs):
        '''
        For evaluating on real data (i.e. no injections.)
        '''
        df = None
        loaded_labels = None
        with h5py.File(self.input_path, 'r') as f:
            print(f.keys())
            encoded_mat = f['sparse_matrix'][0]
            npz_mat = base64.b64decode(encoded_mat)
            sparse_mat = load_npz(io.BytesIO(npz_mat))

            # Convert counts to bools
            if not self.count_concepts:
                sparse_mat = sparse_mat.astype(bool)
            else:
                print('Not converting to bools')

            # if 'coco_2014' in self.input_path:
            if 'dino' in self.input_path or 'k350' in self.input_path or 'k2' in self.input_path:
                # Don't load labels
                df = pd.DataFrame.sparse.from_spmatrix(sparse_mat)
            else:
                loaded_labels = [l.decode('utf-8') for l in list(f['labels'])]
                # df = pd.DataFrame.sparse.from_spmatrix(sparse_mat, columns=loaded_labels)
                df = pd.DataFrame.sparse.from_spmatrix(sparse_mat)
                label_counts = Counter(loaded_labels)
                df_columns = list(loaded_labels)
                duplicate_detected = False
                dropped_indices = set()
                for label, count in label_counts.items():
                    if count > 1:
                        duplicate_detected = True
                        print('WARNING: Duplicate labels found.')
                        label_indices = [i for i, x in enumerate(df_columns) if x == label]
                        col_arr = np.array(df.iloc[:, label_indices])
                        if self.count_concepts:
                            final_column = np.sum(col_arr, axis=1).astype(int)
                        else:
                            final_column = np.any(col_arr, axis=1)
                        final_column = final_column.reshape(*final_column.shape)
                        df.iloc[:, label_indices[0]] = final_column

                        # drop all the rest and pop labels off of loaded_labels
                        for i in label_indices[1:]:
                            dropped_indices.add(i)
                        # Rename the columns from loaded_labels
                            
                df = df.drop(columns=[df.columns[i] for i in dropped_indices])
                loaded_labels = [label for i, label in enumerate(loaded_labels) if i not in dropped_indices]
                df.columns = loaded_labels

                if not duplicate_detected:
                    df.columns = loaded_labels

            self.n_concepts = sparse_mat.shape[1]
            limit = 100_000

            df = df.head(limit)
            if 'cut' in kwargs and kwargs['cut']:
                discretize_by_cut(df, 4, qcut=False)
            elif 'qcut' in kwargs and kwargs['qcut']:
                discretize_by_cut(df, 4, qcut=True)

            self.sparse_mat = sparse_mat

            # Set some cols
            df['gt_label'] = list(f['gt_labels'][:limit])
            df['pred_label'] = list(f['pred_labels'][:limit])
            df['iou'] = list(f['ious'][:limit])

            # Metadata
            df['img_path'] = list(f['img_paths'][:limit])
            df['gt_bbox'] = list(f['gt_bboxes'][:limit])
            df['pred_bbox'] = list(f['pred_bboxes'][:limit])

            if 'crowding' in kwargs and kwargs['crowding']:
                print('doing discretization')
                df['crowding'] = np.maximum(
                    np.array(f['gt_crowds'][:limit]), 
                    np.array(f['pred_crowds'][:limit])
                )
                discretize_col_by_cut(df, 'crowding', 4, qcut=True, fix_range=False)

            # Apply parsing
            # print('Parsing')
            df['gt_bbox'] = df['gt_bbox'].map(parse_bbox_str)
            df['pred_bbox'] = df['pred_bbox'].map(parse_bbox_str)
            if 'bbox_area' in kwargs and kwargs['bbox_area']:
                print('discretization 2!')
                df['gt_bbox_area'] = df['gt_bbox'].map(lambda x: -1 if not len(x) else (x[2] - x[0]) * (x[3] - x[1]))
                df['gt_bbox_area'] = q_cut_percent(df['gt_bbox_area'], 4)

            df['img_path'] = df['img_path'].map(lambda x: x.decode('utf-8'))
            # print('this is wack', df['img_path'].head(5))

            # Print if any gt_label is None
            if df['gt_label'].isnull().values.any():
                print('WARNING: Some ground truth labels are missing.')

            if df['pred_label'].isnull().values.any():
                print('WARNING: Some pred labels are missing.')

        # Trim sparse mat to relevant index columns
        self.sparse_mat = self.sparse_mat[list(df.index)]

        print(df['gt_label'].head(5), 'last print')

        problem = {
            'problem_df': df,
            'slice': [],
            'combo': 3,
            'top_k': 3,
        } 
        yield loaded_labels, problem



    def hdf5_evaluation_iterator_dup(self, limit: Optional[int] = None, **kwargs):
        '''
        For evaluating on real data (i.e. no injections.)
        '''
        df = None
        loaded_labels = None
        dup_count = 0
        cid_labels = None
        merged_cids = {} # map cid -> [all merged cids including itself]
        with h5py.File(self.input_path, 'r') as f:
            # print(f.keys())
            encoded_mat = f['sparse_matrix'][0]
            npz_mat = base64.b64decode(encoded_mat)
            sparse_mat = load_npz(io.BytesIO(npz_mat))

            # Convert counts to bools
            if not self.count_concepts:
                print('Converting to bools')
                sparse_mat = sparse_mat.astype(bool)
            else:
                print('Not converting to bools')

            # if 'coco_2014' in self.input_path:
            if 'dino' in self.input_path or 'k350' in self.input_path or 'k2' in self.input_path:
                # Don't load labels
                df = pd.DataFrame.sparse.from_spmatrix(sparse_mat)
            else:
                loaded_labels = [l.decode('utf-8') for l in list(f['labels'])]
                # df = pd.DataFrame.sparse.from_spmatrix(sparse_mat, columns=loaded_labels)
                df = pd.DataFrame.sparse.from_spmatrix(sparse_mat)
                label_counts = Counter(loaded_labels)
                df_columns = list(loaded_labels)
                cid_labels = list(range(len(loaded_labels)))
                duplicate_detected = False
                dropped_indices = set()
                for label, count in label_counts.items():
                    if count > 1:
                        dup_count += count - 1
                        print('WARNING: Duplicate labels found.')
                        label_indices = [i for i, x in enumerate(df_columns) if x == label]
                        col_arr = np.array(df.iloc[:, label_indices])
                        if self.count_concepts:
                            final_column = np.sum(col_arr, axis=1).astype(int)
                        else:
                            final_column = np.any(col_arr, axis=1).astype(int)

                        merged_cids[label_indices[0]] = label_indices
                        final_column = final_column.reshape(*final_column.shape)
                        df.iloc[:, label_indices[0]] = final_column

                        # drop all the rest and pop labels off of loaded_labels
                        for i in label_indices[1:]:
                            dropped_indices.add(i)
                        
                            
                df = df.drop(columns=[df.columns[i] for i in dropped_indices])
                loaded_labels = [label for i, label in enumerate(loaded_labels) if i not in dropped_indices]
                cid_labels = [label for i, label in enumerate(cid_labels) if i not in dropped_indices]
                df.columns = loaded_labels

                if not duplicate_detected:
                    df.columns = loaded_labels
                        

            self.n_concepts = sparse_mat.shape[1]
            limit = settings.limit_num_rows

            df = df.head(limit)
            if 'cut' in kwargs and kwargs['cut']:
                discretize_by_cut(df, 4, qcut=False)
            elif 'qcut' in kwargs and kwargs['qcut']:
                discretize_by_cut(df, 4, qcut=True)

            self.sparse_mat = sparse_mat

            # Set some cols
            df['gt_label'] = list(f['gt_labels'][:limit])
            df['pred_label'] = list(f['pred_labels'][:limit])
            df['iou'] = list(f['ious'][:limit])

            # Metadata
            df['img_path'] = list(f['img_paths'][:limit])
            df['gt_bbox'] = list(f['gt_bboxes'][:limit])
            df['pred_bbox'] = list(f['pred_bboxes'][:limit])

            if 'crowding' in kwargs and kwargs['crowding']:
                print('doing discretization')
                df['crowding'] = np.maximum(
                    np.array(f['gt_crowds'][:limit]), 
                    np.array(f['pred_crowds'][:limit])
                )
                discretize_col_by_cut(df, 'crowding', 4, qcut=True, fix_range=False)

            # Apply parsing
            df['gt_bbox'] = df['gt_bbox'].map(parse_bbox_str)
            df['pred_bbox'] = df['pred_bbox'].map(parse_bbox_str)
            if 'bbox_area' in kwargs and kwargs['bbox_area']:
                print('discretization 2!')
                df['gt_bbox_area'] = df['gt_bbox'].map(lambda x: -1 if not len(x) else (x[2] - x[0]) * (x[3] - x[1]))
                df['gt_bbox_area'] = q_cut_percent(df['gt_bbox_area'], 4)

            df['img_path'] = df['img_path'].map(lambda x: x.decode('utf-8'))

            # Print if any gt_label is None
            if df['gt_label'].isnull().values.any():
                print('WARNING: Some ground truth labels are missing.')

            if df['pred_label'].isnull().values.any():
                print('WARNING: Some pred labels are missing.')
        # Trim sparse mat to relevant index columns
        self.sparse_mat = self.sparse_mat[list(df.index)]

        problem = {
            'problem_df': df,
            'slice': [],
            'combo': 3,
            'top_k': 3,
        } 
        yield loaded_labels, problem, dup_count, cid_labels, merged_cids


    def get_dataset_iterator(self, limit: Optional[int] = None, **kwargs):
        if self.input_ext == ExtensionTypes.CSV:
            raise NotImplementedError('CSV not implemented yet.')
        elif self.input_ext == ExtensionTypes.HDF5_EVAL:
            return self.hdf5_evaluation_iterator(limit, **kwargs)
        
    def get_dataset_iterator_dup(self, limit: Optional[int] = None, **kwargs):
        if self.input_ext == ExtensionTypes.CSV:
            raise NotImplementedError('CSV not implemented yet.')
        elif self.input_ext == ExtensionTypes.HDF5_EVAL:
            return self.hdf5_evaluation_iterator_dup(limit, **kwargs)

    def get_concept_mapping(self)->dict:
        with h5py.File(self.input_path, 'r') as f:
            if 'concept_mapping' not in f.attrs:
                return None
            return json.loads(f.attrs['concept_mapping'])

    def get_label_mapping(self)->dict:
        with h5py.File(self.input_path, 'r') as f:
            loaded_labels = [l.decode('utf-8') for l in list(f['labels'])]
            return {loaded_labels[i]:i for i in range(len(loaded_labels))}