from bbox_basic_overlap import _compute_ious
from crowd import check_crowding

import pandas as pd
import numpy as np
from tqdm import tqdm

class PairConfidence:
    '''
    Greedy pairing from:
    https://github.com/rafaelpadilla/review_object_detection_metrics/blob/2efe66d2c4b89e4fcc64e490d4caced2096f03aa/src/evaluators/coco_evaluator.py#L303C1-L303C75

    Pair ground-truth and predicted bounding boxes by greedily 
    maximizing confidence first, then IOU.
    '''
    def __init__(self, pickle_data, max_detections, min_iou=0.1, 
                 fp_iou_thresh=0.5, min_confidence=0.5, testing=False,
                 no_union=False):
        self.data = pickle_data
        self.max_detections = max_detections
        self.min_iou = min_iou
        self.fp_iou_thresh = fp_iou_thresh 
        self.min_confidence = min_confidence
        self.no_union = no_union
        self.testing = testing

    def _filter_data(self, data_row):
        '''
        Make sure that bounding boxes meet minimum confidence and sort the
        parameters by confidence.
        '''
        x = data_row
        img_path = x['img_path']
        if 'Images/' in img_path:
            img_path = img_path.split('Images/')[1]
        else:
            img_path = img_path.split('/')[-1]
        pred_instances = x['pred_instances']
        pred_bboxes = pred_instances['bboxes'].numpy()
        pred_labels = pred_instances['labels'].numpy()
        pred_scores = pred_instances['scores'].numpy()

        gt_bboxes = np.array(x['gt_bboxes'].tensor.numpy())
        gt_labels = x['gt_bboxes_labels']

        if 'gt_ignore_flags' in x:
            gt_ignore_flags = x['gt_ignore_flags'].astype(int).tolist()
        elif 'instances' in x and len(x['instances']) > 0 and 'ignore_flag' in x['instances'][0]:
            gt_ignore_flags = [instance['ignore_flag'] for instance in x['instances']]
        else:
            gt_ignore_flags = [0 for _ in range(len(gt_bboxes))]

        # Filter out all predictions < min_confidence
        bbox_mask = pred_scores >= self.min_confidence
        pred_bboxes = pred_bboxes[bbox_mask]
        pred_labels = pred_labels[bbox_mask]
        pred_scores = pred_scores[bbox_mask]
        
        # Sort by highest scores first. 
        dt_sort = np.argsort(-pred_scores, kind="stable")
        pred_bboxes = [pred_bboxes[idx] for idx in dt_sort[:self.max_detections]]
        pred_labels = [pred_labels[idx] for idx in dt_sort[:self.max_detections]]
        pred_scores = [pred_scores[idx] for idx in dt_sort[:self.max_detections]]

        if not self.testing:
            #from mmdet.evaluation import bbox_overlaps as recall_overlaps
            from bbox_basic_overlap import bbox_overlaps as recall_overlaps
            iou_matrix = recall_overlaps(np.array(pred_bboxes), gt_bboxes)
            # iou_matrix = _compute_ious(pred_bboxes, gt_bboxes)
        else:
            iou_matrix = _compute_ious(pred_bboxes, gt_bboxes)

        return (img_path, gt_bboxes, gt_ignore_flags, gt_labels, pred_bboxes, pred_labels, 
                pred_scores, iou_matrix)

    def greedy_pair(self):
        pass

    def pair(self):
        df_data = []
        for x in tqdm(self.data):
            (img_path, gt_bboxes, gt_ignore_flags, gt_labels, pred_bboxes, 
             pred_labels, pred_scores, iou_matrix) = self._filter_data(x)
            
            # Get crowding metrics
            (gt_crowding, pred_crowding, 
             gt_confusion, pred_confusion) = check_crowding(
                pred_bboxes, gt_bboxes, pred_labels, gt_labels, no_union=self.no_union)

            dtm = {}
            gtm = {}
            for d_idx in range(len(pred_bboxes)):
                # information about best match so far (m=-1 -> unmatched)
                iou = min(self.min_iou, 1 - 1e-10)
                m = -1
                for g_idx in range(len(gt_bboxes)):
                    # if this gt already matched, and not a crowd, continue
                    if g_idx in gtm:
                        continue
                    # continue to next gt unless better match made
                    if iou_matrix[d_idx, g_idx] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = iou_matrix[d_idx, g_idx]
                    m = g_idx
                    
                # if match made store id of match for both dt and gt
                if m == -1:
                    continue
                dtm[d_idx] = m
                gtm[m] = d_idx
                
            # FN: GT exists, no match with prediction
            # FP_type1: Pred exists, no matching GT
            # FP_type2: Pred exists, but IOU thresh too low.
            for gt_idx in range(len(gt_bboxes)):
                dt_bbox = np.array([])
                iou, dt_label, score = -1, -1, -1
                fp_type1, fp_type2, fn = False, False, False
                gt_label = gt_labels[gt_idx]
                gt_ignore_flag = gt_ignore_flags[gt_idx]

                # if the gt bbox has been matched, check if the IOU passed threshold
                # if not, then it's a type-2 FP
                crowd_data = [gt_crowding[gt_idx], 0, gt_confusion[gt_idx], 0]
                if gt_idx in gtm:
                    dt_idx = gtm[gt_idx]
                    dt_bbox = pred_bboxes[dt_idx]
                    score = pred_scores[dt_idx]
                    iou = iou_matrix[dt_idx, gt_idx]
                    fp_type2 = iou < self.fp_iou_thresh
                    dt_label = pred_labels[dt_idx]

                    crowd_data[1] = pred_crowding[dt_idx]
                    crowd_data[3] = pred_confusion[dt_idx]
                else:
                    fn = True

                metadata = [img_path]
                bbox_data = [gt_bboxes[gt_idx], dt_bbox, gt_label, dt_label, gt_ignore_flag]    
                metrics = [score, iou, fp_type1, fp_type2, fn]
                row = metadata + bbox_data + metrics + crowd_data

                # row = [img_path, gt_bboxes[gt_idx], dt_bbox, gt_label, dt_label, gt_ignore_flag, score, iou, fp_type1, fp_type2, fn] + crowd_data
                df_data.append(row)

            # Handle FP type1: Prediction exists but never matches with GT
            for dt_idx in range(len(pred_bboxes)):
                if dt_idx not in dtm:
                    crowd_data = [0, pred_crowding[dt_idx], 0, pred_confusion[dt_idx]]
                    gt_bbox = np.array([])
                    dt_bbox = pred_bboxes[dt_idx]
                    gt_label, gt_ignore_flag, iou = -1, -1, -1
                    dt_label = pred_labels[dt_idx]
                    score = pred_scores[dt_idx]
                    fp_type1, fp_type2, fn = True, False, False

                    metadata = [img_path]
                    bbox_data = [gt_bbox, dt_bbox, gt_label, dt_label, gt_ignore_flag]    
                    metrics = [score, iou, fp_type1, fp_type2, fn]
                    row = metadata + bbox_data + metrics + crowd_data
                    
                    # row = [img_path, gt_bbox, dt_bbox, gt_label, dt_label, gt_ignore_flag, score, iou, fp_type1, fp_type2, fn] + crowd_data
                    df_data.append(row)

        df = pd.DataFrame(df_data, columns=['img_path','gt_bbox', 'pred_bbox', 'gt_label', 'pred_label', 'ignore_flag', 'pred_score', 'iou', 'fp_type1', 'fp_type2', 'fn'] +
                          ['gt_crowding', 'pred_crowding', 'gt_confusion', 'pred_confusion'])
        df['gt_bbox'] = df['gt_bbox'].map(lambda x: np.array2string(x, precision=2, separator=','))
        df['pred_bbox'] = df['pred_bbox'].map(lambda x: np.array2string(x, precision=2, separator=','))
        return df
