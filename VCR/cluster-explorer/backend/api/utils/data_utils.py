from dataclasses import dataclass
from io import BytesIO
import base64
import json

from scipy.sparse import save_npz
import h5py

def sparse_to_npz(mat):
    '''
    Converts a sparse matrix to a npz string
    '''
    buffer = BytesIO()
    save_npz(buffer, mat)
    buffer.seek(0)
    npz_str = buffer.getvalue()
    buffer.close()
    return npz_str

@dataclass
class DataPackage:
    img_path: list
    gt_label: list
    pred_label: list
    gt_bbox: list # List of arrays as strings (e.g. ['123,123,123,123',...])
    pred_bbox: list # List of arrays as strings (e.g. ['123,123,123,123',...])
    ignore_flag: list
    pred_score: list
    fp_type1: list
    fp_type2: list
    fn: list
    iou: list

    gt_crowd: list
    pred_crowd: list
    gt_confusion: list
    pred_confusion: list

def save_to_hdf5(out_file, config_str, npz_str:str, labels, 
                 data_pack:DataPackage, concept_mapping:dict=None, interacting:list=None): 
    # Encode again because .decode('utf-8') does not like null-bytes
    # Can possibly use .decode('latin-1'), but seems sketchy
    encoded_npz = base64.b64encode(npz_str)
    g = 'gzip'
    #print('THis is ignore flag', data_pack.ignore_flag)
    with h5py.File('{}.hdf5'.format(out_file), 'w') as f:
        f.create_dataset('sparse_matrix', data=[encoded_npz]) # Must be in array
        f.create_dataset('labels', data=labels, compression=g)
        f.create_dataset('img_paths', data=data_pack.img_path, compression=g)
        f.create_dataset('gt_labels', data=data_pack.gt_label, compression=g)
        f.create_dataset('pred_labels', data=data_pack.pred_label, compression=g)
        f.create_dataset('gt_bboxes', data=data_pack.gt_bbox, compression=g)
        f.create_dataset('pred_bboxes', data=data_pack.pred_bbox, compression=g)
        f.create_dataset('ignore_flags', data=data_pack.ignore_flag, compression=g)
        f.create_dataset('pred_scores', data=data_pack.pred_score, compression=g)
        f.create_dataset('fp_type1s', data=data_pack.fp_type1, compression=g)
        f.create_dataset('fp_type2s', data=data_pack.fp_type2, compression=g)
        f.create_dataset('fns', data=data_pack.fn, compression=g)
        f.create_dataset('ious', data=data_pack.iou, compression=g)

        f.create_dataset('gt_crowds', data=data_pack.gt_crowd, compression=g)
        f.create_dataset('pred_crowds', data=data_pack.pred_crowd, compression=g)
        f.create_dataset('gt_confusions', data=data_pack.gt_confusion, compression=g)
        f.create_dataset('pred_confusions', data=data_pack.pred_confusion, compression=g)
        f.attrs['config'] = config_str

        if interacting:
            print('interacting created')
            f.create_dataset('interacting', data=interacting, compression=g)

        if concept_mapping:
            concept_mapping_json = json.dumps(concept_mapping)
            f.attrs['concept_mapping'] = concept_mapping_json