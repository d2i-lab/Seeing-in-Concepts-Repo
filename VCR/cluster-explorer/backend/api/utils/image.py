import api.utils.query as query
import os
import matplotlib.pyplot as plt 
import cv2
import pycocotools
import json
import numpy as np
import hashlib
import matplotlib

matplotlib.use('Agg')


import api.utils.settings as settings

settings = settings.Settings()

concept_json_path = settings.sam_jsons_dir
img_folder_path = settings.coco_img_dir
img_label_path = settings.label_dir

with open(img_label_path, 'r') as f:
    coco_labels = f.readlines()
    coco_labels = [label.strip() for label in coco_labels]

def calculate_union(box1, box2):
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])
    
    union_box = [x_min, y_min, x_max, y_max]
    return union_box

def matrix_subarray(full_matrix, bbox, scale=.1):
    h = bbox[3] - bbox[1]
    w = bbox[2] - bbox[0]
    padding_row = int(w * scale/2)
    padding_col = int(h * scale/2)
    row1 = max(int(bbox[1])-padding_row, 0)
    row2 = min(int(bbox[3])+padding_row, full_matrix.shape[0])
    col1 = max(int(bbox[0])-padding_col, 0)
    col2 = min(int(bbox[2])+padding_col, full_matrix.shape[1])
    img_bbox = full_matrix[row1:row2,col1:col2]
    return img_bbox

def draw_bbox(img, bbox, bbox_color, bbox_label, crowding=None):
    bbox = bbox.astype(int)
    bbox_label = coco_labels[bbox_label] 
    if crowding:
        bbox_label = bbox_label + ' ({})'.format(crowding)
    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
    img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), bbox_color, 2)
    
    # Text and background
    (w, h), _ = cv2.getTextSize(
        bbox_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    
    text_color = (255,255,255)
    img = cv2.rectangle(img, (x_min, y_min - 20), (x_min + w, y_min), bbox_color, -1)
    img = cv2.putText(img, bbox_label, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
 
    return img

def extract_sam_mask(json_file):
    '''
    Extract SAM masks.
    '''
    with open(json_file, 'r') as f:
        json_dict = json.load(f)
    segments = [seg['segmentation'] for seg in json_dict]
    decoded_segments = []
    for seg in segments:
        decoded_segments.append(pycocotools.mask.decode(seg))

    return decoded_segments

def one_percent_filter_classic(segments):
    '''
    Method to filter out segments that are less than 1% of the image.
    * Remove all segments < 10 pixels 
    * Then return the indices of segments that are >= 1% of the image

    The multiple passes through the data is inefficient, but because of
    how we structured the data the first time, we have to follow the
    filtering exactly as it was done the first time.
    '''
    segment_sizes = [np.sum(seg) for seg in segments]
    segments = [seg for seg, size in zip(segments, segment_sizes) if size > 10]

    with_id = []
    mask_size_threshold = segments[0].size * 0.01
    for i, seg in enumerate(segments):
        if segment_sizes[i] >= mask_size_threshold:
            with_id.append((i, seg))
    return with_id

def draw_concepts_sam(img_pixels, json_path, gt_bbox, pred_bbox, mapping, concept_ids, cid_to_label_map):
    colors = [
        (175, 25, 0),
        (100, 100, 0), 
        (0, 150, 85), 
        (0, 55, 150), 
        (150, 125, 125),
        (0, 125, 255),
        (150, 150, 0),
        (80, 155, 25),
        (80, 20, 150),
    ]
    segments = extract_sam_mask(json_path)
    to_display = []

    # Needed mapping: concept_id->[seg_id]
    segment_to_cid = {}
    for cid in concept_ids:
        if cid not in mapping:
            print('Missing', cid)
            continue
        for segment_id in mapping[cid]:
            to_display.append(int(segment_id) - 1)
            segment_to_cid[int(segment_id) - 1] = cid

    to_display = list(set(to_display))
    labels = []
    canvas = np.zeros_like(img_pixels)
    new_to_display = []

    for seg_id in to_display:
        label = segment_to_cid[seg_id]
        label = cid_to_label_map[label]
        new_to_display.append((segments[seg_id], str(label)))


    for idx, (mask, label) in enumerate(new_to_display):
        c = colors[idx%len(colors)]
        bool_mask = mask == 1
        canvas[bool_mask > 0] = c
        indices = np.argwhere(mask == 1)
        average_position = np.median(indices, axis=0).astype(int)
        labels.append((label, average_position))

        # Optional: Add contours to display images (slow)
        # canvas_converted = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        # contours, _ = cv2.findContours(canvas_converted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(canvas, contours, -1, (255, 255, 255), 2, lineType=cv2.LINE_4)
    
    canvas = canvas.astype(int) * .55 + img_pixels * .45

    for (label, average_position) in labels:
        text_color = (255,255,255)
        label = str(label)
        canvas = cv2.putText(canvas, label, (average_position[1], average_position[0] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        
    return canvas.astype(int)

# def reveal_concepts(df, title_prefix, img_path, gt_label, pred_label, gt_bbox, pred_bbox, gt_crowding,pred_crowding):
def reveal_concepts(dir, df, title_prefix, img_path, gt_label, pred_label, gt_bbox, pred_bbox, cids, mapping, cid_to_label_map, draw_all_bbox=False, facecolor=None):
    '''
    cids: list of concept ids
    mapping: dict of concept id -> seg id
    cid_to_label_map: dict of concept id -> label
    '''
    # json_name = '{}-masks.json'.format(img_path)
    json_name = '{}.json'.format(img_path)
    concept_json = os.path.join(concept_json_path, json_name)
    if not os.path.isfile(concept_json):
        json_name = '{}.json'.format(img_path)
        concept_json = os.path.join(concept_json_path, json_name)

    if not os.path.isfile(concept_json):
        print('No concept json found for {}'.format(img_path))
        print(concept_json)
        return None

    img_path_original = os.path.join(img_folder_path, img_path)
    img_pixels = cv2.imread(img_path_original)
    img_pixels = cv2.cvtColor(img_pixels, cv2.COLOR_BGR2RGB)
    
    # plot all other bboxes
    img_path = img_path_original.split('/')[-1]

    if draw_all_bbox:
        same_pic_df = df[(df['img_path']==img_path)]

        for pic_gt_label, pic_gt_bbox in zip(same_pic_df['gt_label'], same_pic_df['gt_bbox']):
            img_pixels = draw_bbox(img_pixels, pic_gt_bbox, (214, 255, 218), pic_gt_label)
            
        for pic_gt_label, pic_gt_bbox in zip(same_pic_df['pred_label'], same_pic_df['pred_bbox']):
            if len(pic_gt_bbox) == 0:
                continue
            img_pixels = draw_bbox(img_pixels, pic_gt_bbox, (138, 198, 255), pic_gt_label)

    fig, axes = plt.subplots(2, 2, facecolor=facecolor)

    concepts_drawn = draw_concepts_sam(
        img_pixels, concept_json, gt_bbox, pred_bbox, mapping, cids, cid_to_label_map)
    img_pixels = draw_bbox(img_pixels, gt_bbox, (0,255,0), gt_label)

    if len(pred_bbox) > 0:
        img_pixels = draw_bbox(img_pixels, pred_bbox, (0,0,255), pred_label)

    
    axes[0, 0].imshow(cv2.cvtColor(cv2.imread(img_path_original), cv2.COLOR_BGR2RGB).astype(int))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_pixels.astype(int))
    axes[0, 1].set_title('BBOX')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(concepts_drawn)
    axes[1, 0].set_title('Concepts')
    axes[1, 0].axis('off')

    if len(pred_bbox) == 0:
        pred_bbox = gt_bbox
    
    bbox_union = calculate_union(gt_bbox, pred_bbox)
    img_pixels = matrix_subarray(concepts_drawn, bbox_union, scale=.45)
    axes[1, 1].imshow(img_pixels.astype(int))
    axes[1, 1].set_title('Zoom')
    axes[1, 1].axis('off')

    title = '{} | img={}'.format(title_prefix, img_path.split('/')[-1])
    
    plt.suptitle(title, fontsize=14)

    hash_input = str(img_path) + str(gt_label) + str(pred_label) + str(gt_bbox) + str(pred_bbox) + str(cids)
    out_name = '{}_{}.png'.format(img_path.split('/')[-1], hashlib.md5(hash_input.encode()).hexdigest())

    fig.tight_layout()
    current_width, current_height = fig.get_size_inches()
    scale_factor = 1.25
    new_width = current_width * scale_factor
    new_height = current_height * scale_factor

    # Set the figure size with the new size
    fig.set_size_inches(new_width, new_height)
    fig.savefig(os.path.join(dir, out_name), bbox_inches='tight')
    plt.close() # Does this close the figure?
    return out_name
        
def visualize_scenes_list(dir, df, slice_, concept_mapping, n_concepts, dummy_to_cid, examples_per=16):
    print('Running vis list. prior to getting slice len', len(df))
    df_sliced = query.get_slice(df, slice_)
    df_sliced = df_sliced.sample(frac=1)
    print('This is df_sliced len', len(df_sliced))
    correct = df_sliced.sort_values(by='iou', ascending=False)
    wrong = correct[::-1]

    cid_to_label_map = {}
    for dummy_name, cid_list in dummy_to_cid.items():
        for cid in cid_list:
            cid_to_label_map[cid] = dummy_name.split('=')[0]
    
    img_name_list = []
    def _is_valid_presence_col(col):
        ignore_prefix = ['crowding', 'gt-bbox-area']
        has_ignore_prefix = any([col.startswith(prefix) for prefix in ignore_prefix])
        if has_ignore_prefix:
            return False
        range_chars = ['(', ')', '[', ']']
        has_range = any([char in col.split('=')[-1] for char in range_chars])
        if has_range:
            return True
        return (col.endswith('=1') or col.endswith('=True'))

    def _get_cids(row, df_cols, slice_, all_concepts=False):
        '''
        Retrieve concepts marked as present.
        df_cols: list of columns to check. Passed in because iterrows() returns awkward format.
        slice_: slice string
        all_concepts: If True, return all concepts. If False, return only concepts in slice.
        '''
        if all_concepts:
            presence_cols = [col for col in df_cols if _is_valid_presence_col(col)]
            col_to_concept = {col: dummy_to_cid[col] for col in presence_cols}
            return [col_to_concept[col] for col in presence_cols if (row[col] == 1 or row[col] == True)]

        items = []


        for item in query.parse_slice(slice_):
            if _is_valid_presence_col(item):
                items.extend(dummy_to_cid[item])

        return items


    # Display lower IOU first
    count = 0
    for i, row in wrong.iterrows():

        if count >= examples_per:
            break
        # print(f'{row=}')
        pred_label_str = coco_labels[row.pred_label]
        gt_label_str = coco_labels[row.gt_label]
        iou_value = round(row.iou, 3)
        text = 'iou={} | (gt-class={}, pred-class={})'.format(iou_value, gt_label_str, pred_label_str)
        concept_to_seg = concept_mapping[row.img_path.split('.')[0]]
        cids = _get_cids(row, df_sliced.columns, slice_)
        color = '#e3ffea'
        if row.iou < 0.5:
            color = '#ffeaeb'
        x = reveal_concepts(dir, df, text, row.img_path, row.gt_label,
                            row.pred_label, row.gt_bbox, row.pred_bbox, cids,
                            concept_to_seg, cid_to_label_map, 
                            facecolor=color)
        img_name_list.append(x)
        count += 1


    count = 0
    for i, row in correct.iterrows():
        if count >= examples_per:
            break

        pred_label_str = coco_labels[row.pred_label]
        gt_label_str = coco_labels[row.gt_label]
        iou_value = round(row.iou, 3)
        text = 'iou={} | (gt-class={}, pred-class={})'.format(iou_value, gt_label_str, pred_label_str)
        concept_to_seg = concept_mapping[row.img_path.split('.')[0]]
        cids = _get_cids(row, df_sliced.columns, slice_)
        color = '#e3ffea'
        if row.iou < 0.5:
            color = '#ffeaeb'
        x = reveal_concepts(dir, df, text, row.img_path, row.gt_label,
                            row.pred_label, row.gt_bbox, row.pred_bbox, cids,
                            concept_to_seg, cid_to_label_map,
                            facecolor=color)
        img_name_list.append(x)
        count += 1
    

    return img_name_list