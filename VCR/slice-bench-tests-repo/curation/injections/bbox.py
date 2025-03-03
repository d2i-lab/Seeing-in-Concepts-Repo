import random

import numpy as np

def calculate_box_area(data):
    bbox_xywh = data.bbox
    w, h = bbox_xywh[2], bbox_xywh[3]
    return w * h

def calculate_box_aspect_ratio(data):
    bbox_xywh = data.bbox
    w, h = bbox_xywh[2], bbox_xywh[3]
    return w / h

def split_into_quantiles(data, n, key=calculate_box_area):
    sorted_data = sorted(data, key=key)
    sort_mapping = {}

    for i, d in enumerate(data):
        box = d.bbox
        sort_mapping[tuple(box)] = i

    chunk_size = len(data) // n
    labels = [-1] * len(data)

    # Map label to bbox size
    # label_map = {}

    label_counter = 0
    for i in range(0, len(sorted_data), chunk_size):
        quantile = sorted_data[i:i+chunk_size]
        for q in quantile:
            labels[sort_mapping[tuple(q.bbox)]] = label_counter
        label_counter += 1

    return labels

# TODO: Iffy--slice buckets can be large.
def get_direction_labels(data):
    def _get_box_center(box_xywh):
        x, y, w, h = box_xywh
        return np.array([x + w/2, y + h/2])

    assert hasattr(data[0], 'pred_bbox'), 'Data must have pred_bbox attribute'
    
    relative_directions = []
    for d in data:
        relative_directions.append(
            _get_box_center(d.pred_bbox) - _get_box_center(d.gt_bbox)
        )

    direction_labels = []
    for v in relative_directions:
        max_component = np.argmax(np.abs(v))
        if max_component == 0:
            direction_labels.append('r' if v[0] > 0 else 'l')
        else:
            direction_labels.append('d' if v[1] > 0 else 'u')

    return direction_labels

def area_by_five(data):
    return split_into_quantiles(data, 5, key=calculate_box_area)

def area_by_ten(data):
    return split_into_quantiles(data, 10, key=calculate_box_area)

def area_by_range(data):
    n_bins = random.randint(5, 20)
    print('area n_bins:', n_bins)
    return split_into_quantiles(data, n_bins, key=calculate_box_area)

def aspect_ratio_by_range(data):
    n_bins = random.randint(5, 20)
    return split_into_quantiles(data, n_bins, key=calculate_box_aspect_ratio)

def aspect_ratio_by_five(data):
    return split_into_quantiles(data, 5, key=calculate_box_aspect_ratio)

def aspect_ratio_by_ten(data):
    return split_into_quantiles(data, 10, key=calculate_box_aspect_ratio)

def get_box_plugin():
    return {
        'box_area5': area_by_five,
        'box_area10': area_by_ten,
        'box_area_range': area_by_range,
        'box_aspect5': aspect_ratio_by_five,
        'box_aspect10': aspect_ratio_by_ten,
        'box_aspect_range': aspect_ratio_by_range,
        # 'box_direction': get_direction_labels
    }