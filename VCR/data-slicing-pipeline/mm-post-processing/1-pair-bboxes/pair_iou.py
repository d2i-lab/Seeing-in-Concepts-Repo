from bbox_basic_overlap import _compute_ious

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

class PairIOU:
    '''
    Maximize SUM of IOU between paired bboxes
    '''
    def __init__(self, pickle_data, min_iou, fp_iou_thresh=0.5, 
                 min_confidence=0.5, testing=False) -> None:
        self.data = pickle_data
        self.min_iou = min_iou
        self.fp_iou_thresh = fp_iou_thresh 
        self.min_confidence = min_confidence
        self.testing = testing

    def _filter_data(self, data_row):
        '''
        Make sure that bounding boxes meet minimum confidence and sort the
        parameters by confidence.
        '''
        x = data_row
        img_path = x['img_path']
        img_path = img_path.split('Images/')[1]
        pred_instances = x['pred_instances']
        pred_bboxes = pred_instances['bboxes'].numpy()
        pred_labels = pred_instances['labels'].numpy()
        pred_scores = pred_instances['scores'].numpy()
        
        gt_bboxes = np.array(x['gt_bboxes'].tensor.numpy())
        gt_labels = x['gt_bboxes_labels']

        # Filter out all predictions < min_confidence
        bbox_mask = pred_scores >= self.min_confidence
        pred_bboxes = pred_bboxes[bbox_mask]
        pred_labels = pred_labels[bbox_mask]
        pred_scores = pred_scores[bbox_mask]

        if not self.testing:
            from mmdet.evaluation import bbox_overlaps as recall_overlaps
            iou_matrix = recall_overlaps(np.array(pred_bboxes), gt_bboxes)
        else:
            iou_matrix = _compute_ious(pred_bboxes, gt_bboxes)
        return (img_path, gt_bboxes, gt_labels, pred_bboxes, pred_labels, 
                pred_scores, iou_matrix)

    def pair(self):
        # Use scipy linear_sum_assignment to find the best pairing
        df_data = []
        for x in tqdm(self.data):
            (img_path, gt_bboxes, gt_labels, pred_bboxes, 
             pred_labels, pred_scores, iou_matrix) = self._filter_data(x)

            # Run linear sum assignment
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            # row_ind is the index of pred_bboxes
            # col_ind is the index of gt_bboxes

            # Create a dictionary of the pairings
            dtm = {}
            gtm = {}
            for d_idx, g_idx in zip(row_ind, col_ind):
                dtm[d_idx] = g_idx
                gtm[g_idx] = d_idx

            # FN: GT exists, no match with prediction
            # FP_type1: Pred exists, no matching GT
            # FP_type2: Pred exists, but IOU thresh too low.
            for gt_idx in range(len(gt_bboxes)):
                dt_bbox = np.array([])
                iou, dt_label, score = -1, -1, -1
                fp_type1, fp_type2, fn = False, False, False
                gt_label = gt_labels[gt_idx]
                if gt_idx in gtm:
                    dt_idx = gtm[gt_idx]
                    dt_bbox = pred_bboxes[dt_idx]
                    score = pred_scores[dt_idx]
                    iou = iou_matrix[dt_idx, gt_idx]
                    dt_label = pred_labels[dt_idx]

                    if iou > 0:
                        fp_type2 = iou < self.fp_iou_thresh
                    else:
                        fn = True
                        dt_bbox, dt_label = np.array([]), -1
                else:
                    fn = True

                row = [img_path, gt_bboxes[gt_idx], dt_bbox, gt_label, dt_label, score, iou, fp_type1, fp_type2, fn]
                df_data.append(row)

            # Handle FP type1: Prediction exists but never matches with GT
            for dt_idx in range(len(pred_bboxes)):
                if dt_idx not in dtm:
                    gt_bbox = np.array([])
                    dt_bbox = pred_bboxes[dt_idx]
                    gt_label, iou = -1, -1
                    dt_label = pred_labels[dt_idx]
                    score = pred_scores[dt_idx]
                    fp_type1, fp_type2, fn = True, False, False
                    
                    row = [img_path, gt_bbox, dt_bbox, gt_label, dt_label, score, iou, fp_type1, fp_type2, fn]
                    df_data.append(row)

        df = pd.DataFrame(df_data, columns=['img_path','gt_bbox', 'pred_bbox', 'gt_label', 'pred_label', 'pred_score', 'iou', 'fp_type1', 'fp_type2', 'fn'])
        df['gt_bbox'] = df['gt_bbox'].map(lambda x: np.array2string(x, precision=2, separator=','))
        df['pred_bbox'] = df['pred_bbox'].map(lambda x: np.array2string(x, precision=2, separator=','))
        return df

            