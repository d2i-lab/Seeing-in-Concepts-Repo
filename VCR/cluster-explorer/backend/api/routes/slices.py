import os
import warnings

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

import api.utils.settings as settings
import api.utils.data as data
import api.models.slice as slice
import api.utils.image as image

from divexplorer3.divexplorer.FP_DivergenceExplorer import FP_DivergenceExplorer
from divexplorer3.divexplorer.FP_Divergence import FP_Divergence


settings = settings.Settings()
router = APIRouter()
loaded_files = []
result_cache = []
result_cache_hash = {}
img_cache = {}

def add_to_cache(request, result):
    global result_cache
    rhash = hash(request)
    if len(result_cache) >= settings.result_cache_size:
        to_remove = result_cache.pop(0)
        del result_cache_hash[to_remove]

    result_cache.append(rhash)
    result_cache_hash[rhash] = result

def check_cache(request):
    global result_cache
    global result_cache_hash


    rhash = hash(request)
    if rhash in result_cache_hash:
        return result_cache_hash[rhash]
    
    return None

def load_file_dup_check(file_name: str, limit: int = 50000, qcut=False, cut=False, crowding=False, bbox_area=False, count_concepts=True):
    global loaded_files

    # TODO: Add back caching
    # Rn, it just loads the file.
    for file in loaded_files:
        # if file.file_name == file_name and file.limit == limit:
        if (
            file.file_name == file_name and
            file.qcut == qcut and
            file.cut == cut and
            file.crowding == crowding and
            file.bbox_area == bbox_area and
            file.count_concepts == count_concepts
        ):
            # print('cache hit', file.qcut, file.dup_count, file.cid_labels)
            return file, file.dup_count, file.cid_labels, file.merged_cids

    print('cache miss')

    full_file_path = os.path.join(settings.data_dir, file_name)
    ext = data.ExtensionTypes.HDF5_EVAL
    dataset = data.Dataset(input_path=full_file_path, input_ext=ext, count_concepts=count_concepts)
    # Ugly hack
    df = None

    # Ideally should just have a single problem df
    # for _, v in dataset.get_dataset_iterator(limit, qcut=qcut, cut=cut, crowding=crowding, bbox_area=bbox_area):
    dup_count = 0
    cid_labels = None
    merged_cids = {}
    for (_, v, dup_count, cid_labels, merged_cids) in dataset.get_dataset_iterator_dup(limit, qcut=qcut, cut=cut, crowding=crowding, bbox_area=bbox_area):
        df = v['problem_df']
        dup_count = dup_count
        cid_labels = cid_labels
        merged_cids = merged_cids 


    print('Data loaded!')
    if len(loaded_files) >= settings.max_files:
        loaded_files.pop(0)

    file = data.LoadedDataset(
        df=df,
        file_name=file_name,
        limit=limit,
        concept_mapping=dataset.get_concept_mapping(),
        label_mapping=dataset.get_label_mapping(),
        n_concepts=dataset.n_concepts,
        count_concepts=count_concepts,
        crowding=crowding,
        bbox_area=bbox_area,
        qcut=qcut,
        cut=cut,
        dup_count=dup_count,
        cid_labels=cid_labels,
        merged_cids=merged_cids
    )
    loaded_files.append(
        file
    )
    # return df
    return file, dup_count, cid_labels, merged_cids


def load_file(file_name: str, limit: int = 50000, qcut=False, cut=False, crowding=False, bbox_area=False, count_concepts=True):
    global loaded_files

    for file in loaded_files:
        if (
            file.file_name == file_name and
            file.qcut == qcut and
            file.cut == cut and
            file.crowding == crowding and
            file.bbox_area == bbox_area and
            file.count_concepts == count_concepts
        ):
            print('cache hit')
            return file

    full_file_path = os.path.join(settings.data_dir, file_name)
    ext = data.ExtensionTypes.HDF5_EVAL
    dataset = data.Dataset(input_path=full_file_path, input_ext=ext, count_concepts=count_concepts)
    # Ugly hack
    df = None

    dup_count = 0
    cid_labels = None
    merged_cids = {}
    for (_, v, dup_count, cid_labels, merged_cids) in dataset.get_dataset_iterator_dup(limit, qcut=qcut, cut=cut, crowding=crowding, bbox_area=bbox_area):
        df = v['problem_df']
        dup_count = dup_count
        cid_labels = cid_labels
        merged_cids = merged_cids

    print('Data loaded!')
    if len(loaded_files) >= settings.max_files:
        loaded_files.pop(0)

    file = data.LoadedDataset(
        df=df,
        file_name=file_name,
        limit=limit,
        concept_mapping=dataset.get_concept_mapping(),
        label_mapping=dataset.get_label_mapping(),
        n_concepts=dataset.n_concepts,
        count_concepts=count_concepts,
        crowding=crowding,
        bbox_area=bbox_area,
        qcut=qcut,
        cut=cut,
        dup_count=dup_count,
        cid_labels=cid_labels,
        merged_cids=merged_cids
    )
    loaded_files.append(
        file
    )
    return file

@router.get('/dir')
def get_dir():
    return list(sorted(os.listdir(settings.data_dir)))

@router.get('/label_choices')
def get_label_choices():
    with open(settings.label_dir, 'r') as f:
        class_labels = f.read().splitlines()
        return class_labels


def apply_error_type(df, error_type):
    if error_type == slice.ErrorTypes.IOU_FP:
        df = df[df['gt_label'] == df['pred_label']]
        df = df[df['iou'] < 0.6]
    elif error_type == slice.ErrorTypes.CLASS_FP:
        df = df[df['gt_label'] != -1]
        df = df[df['pred_label'] != -1]
        df = df[df['gt_label'] != df['pred_label']]

    return df  

def apply_label(df, label, both=True):
    if label is not None:
        df = df[df['gt_label'] == label]
        df = df[df['gt_label'] == df['pred_label']]

    return df

def apply_all(request, current_file, keep=[]):
    current_file_copy = current_file.copy(deep=True)

    if request.label is not None and request.label >= 0:
        current_file_copy = apply_label(current_file_copy, request.label)

    always_drop = set(['gt_label', 'pred_label', 'iou', 'img_path', 'gt_bbox', 'pred_bbox']).difference(set(keep))
    to_drop = list(filter(lambda x: x in current_file_copy.columns, always_drop))

    current_file_copy['Outlier'] = (current_file_copy['iou'] < 0.6).astype(bool)
    current_file_copy = current_file_copy[current_file_copy['iou'] >= 0]
    current_file_copy = current_file_copy.drop(columns=to_drop)
    return current_file_copy

def divexplore_fit(dataframe, min_support, max_len, th_redundancy=None, include_absence=True, FPM_type='fpgrowth'):
    """
    Run divexplorer given dataframe and other params.
    
    Returns:
    - result_divexplore: Dataframe with itemsets and other info.
    - fp_divergence_acc: Data containing post-mining results.
    """
    negative = 0
    positive = 1
    true_class, predicted_class = 'true', 'predicted'
    df = dataframe.copy()

    fp_diver = FP_DivergenceExplorer(df, true_class, predicted_class, class_map={"P":positive, "N":negative}, include_absence=include_absence)
    FP_fm = fp_diver.getFrequentPatternDivergence(min_support=min_support, metrics=["d_accuracy"],max_len=max_len, FPM_type=FPM_type)
     
    # Fitler on d_accuracy and run de-dup functions (if th_redundancy != None)
    result_divexplore=FP_fm
    fp_divergence_acc=FP_Divergence(FP_fm, "d_accuracy")
    result_divexplore=fp_divergence_acc.getDivergence(th_redundancy=th_redundancy)
    result_divexplore = result_divexplore.sort_values(by=['d_accuracy'], ascending=True)
    return result_divexplore, fp_divergence_acc


def start_mining(request: slice.MineRequest):
    global loaded_files

    file_name = request.file
    full_path = os.path.join(settings.data_dir, file_name)
    print('Mining request', request)
    if os.path.isfile(full_path) and file_name.endswith('.hdf5'):
        current_file = load_file(file_name, request.limit, cut=request.cut, qcut=request.qcut, crowding=request.crowding, bbox_area=request.bbox_area, count_concepts=request.count)
    else:
        raise HTTPException(status_code=404, detail='File not found')

    if current_file is None:
        raise HTTPException(status_code=404, detail='File not found')

    df = current_file.df
    df_copy = apply_all(request, df)
    df_copy['predicted'] = ~df_copy['Outlier']
    df_copy['true'] = True
    df_copy = df_copy.drop(columns=['Outlier'])

    divexp_results, _ = divexplore_fit(df_copy, request.support, max_len=request.max_combo, th_redundancy=None,
                              include_absence=False)
    results = divexp_results.head(request.top_k).copy()
    results['itemsets'] = results['itemsets'].apply(lambda x: tuple(x))
    results = results.rename(columns={'support_count': 's_count'})
    results = results[['itemsets', 's_count', 'support', 'accuracy', 'd_accuracy']]
    result_json = results.to_json(orient='records')
    add_to_cache(request, result_json)
    return result_json


@router.post('/sync-mine')
def request_sync_mine(request: slice.MineRequest):
    cache_result = check_cache(request)
    if cache_result != None:
        return cache_result

    return start_mining(request)

@router.get('/list-loaded')
def request_list_loaded():
    global loaded_files
    return [(lf.file_name, lf.limit) for lf in loaded_files]

@router.get('/img/{file_name}')
def request_get_img(file_name: str):
    file_name = os.path.join(settings.img_dir, file_name)
    if os.path.exists(file_name) and os.path.abspath(file_name).endswith('.png'):
        return FileResponse(file_name)
    return HTTPException(status_code=404, detail='File not found')

@router.post('/visualize2')
def get_slice_visualize(request: slice.VisRequest):
    global loaded_files
    global img_cache

    if len(img_cache) > settings.img_cache_size:
        random_key = list(img_cache.keys())[0]
        del img_cache[random_key]

    if request in img_cache:
        return img_cache[request]
    
    print('Visualization request')
    file_name = request.file
    full_path = os.path.join(settings.data_dir, file_name)
    if os.path.isfile(full_path) and file_name.endswith('.hdf5'):
        current_file, dup_count, cid_labels, merged_cids = load_file_dup_check(file_name, request.limit, cut=request.cut, qcut=request.qcut, crowding=request.crowding, bbox_area=request.bbox_area)
    else:
        raise HTTPException(status_code=404, detail='File not found')


    # Create img -> {concept id -> [segment idx]}
    concept_mapping = current_file.concept_mapping # concept -> {img -> segment idx}g
    n_concepts = current_file.n_concepts
    n_concepts -= dup_count
    concept_mapping = data.get_concept_per_image(concept_mapping)
    
    # Relabel concept columns back to their original id's
    df_copy = current_file.df.copy()

    cols_to_keep = ['iou', 'img_path', 'gt_bbox', 'pred_bbox', 'gt_label', 'pred_label']
    df_copy = apply_all(request, df_copy, keep=cols_to_keep)
    original_columns = list(df_copy.columns)
    dummy_cols = list(df_copy.columns[:n_concepts])

    if request.bbox_area:
        dummy_cols.append('gt_bbox_area')
    if request.crowding:
        dummy_cols.append('crowding')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df_copy = pd.get_dummies(df_copy, prefix_sep='=', columns=dummy_cols)

    
    # Maps column names to their original concept id
    col_to_cid = {col: i for i, col in zip(cid_labels, original_columns)}
    dummy_to_cid = {}

    for col in df_copy.columns:
        col_name = col.split('=')[0]
        if col_name in col_to_cid:
            cid = col_to_cid[col_name]
            if cid in merged_cids:
                cid = merged_cids[cid]
            else:
                cid = [cid]
            dummy_to_cid[col] = cid

    img_names = image.visualize_scenes_list(settings.img_dir, df_copy, request.slice, concept_mapping, current_file.n_concepts, dummy_to_cid, examples_per=5)
    img_cache[request] = img_names
    return img_names


@router.get('/reset_cache')
def reset_cache():
    global result_cache
    global result_cache_hash
    global img_cache
    global loaded_files

    result_cache = []
    result_cache_hash = {}
    img_cache = {}
    loaded_files = []
    return {}
