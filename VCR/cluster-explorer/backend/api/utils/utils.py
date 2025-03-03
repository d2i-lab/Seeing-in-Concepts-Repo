import json
import pycocotools
import numpy as np

def parse_bbox_str(bbox_str, dtype=float):
   return np.fromstring(bbox_str[1:-1], dtype=dtype, sep=',') 

def get_segment_concept_per_image(remap):
    '''
    Given a remap, return a dictionary mapping image names to a dictionary
    mapping segment ids to concept ids.
    '''
    img_to_mapping = {} # Img Name -> {Seg ID -> Concept ID}
    for cluster_id in remap['cluster_map']:
        for img_name in remap['cluster_map'][cluster_id]:
            if img_name not in img_to_mapping:
                img_to_mapping[img_name] = {}
            for seg_id in remap['cluster_map'][cluster_id][img_name]:
                img_to_mapping[img_name][seg_id] = int(cluster_id)

    return img_to_mapping

def extract_segments(json_file):
    with open(json_file, 'r') as f:
        json_dict = json.load(f)
    segments = [seg['segmentation'] for seg in json_dict]
    decoded_segments = []
    for seg in segments:
        decoded_segments.append(pycocotools.mask.decode(seg))

    return decoded_segments