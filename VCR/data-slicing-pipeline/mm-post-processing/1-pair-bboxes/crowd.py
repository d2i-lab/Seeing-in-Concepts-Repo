from bbox_basic_overlap import _compute_ious, get_bbox_area

def is_crowded(pred_bbox, gt_bbox, mat, iou_thresh=0.1, no_union=True):
    if no_union:
        pred_thresh = get_bbox_area(pred_bbox) * iou_thresh
        gt_thresh = get_bbox_area(gt_bbox) * iou_thresh

        return mat >= min(pred_thresh, gt_thresh) 


    return mat >= iou_thresh



def check_crowding(pred_bboxes, gt_bboxes, pred_labels, gt_labels, iou_thresh=0.1,
                   no_union=True):
    '''
    For each bounding box, determine the number of overlapping boudning boxes.
    Furthermore, coutn the number of overlapping bounding boxes with same 
    labels (confusion).

    Args:
        pred_bboxes: List of predicted bounding boxes
        gt_bboxes: List of ground-truth bounding boxes
        pred_labels: Labels corresponding to prediction bounding boxes
        gt_labels: Labels corresponding to ground-truth bounding boxes
        iou_thresh: Minimum iou needed to count as an intersection

    Returns:
        gt_crowding: A dictionary mapping ground-truth bounding box indices to
          the number of intersections with prediction bounding boxes
        pred_crowding: A dictionary mapping prediction bounding box indices to
          the number of intersections with ground-truth bounding boxes
        gt_confusion: Same as gt_crowding except with the requiremnt that the
          labels match.
        pred_confusion: Same as pred_crowding except with the requiremnt that 
          the labels match.
    '''
    gt_crowding = {g_idx:0 for g_idx in range(len(gt_bboxes))}
    pred_crowding = {d_idx:0 for d_idx in range(len(pred_bboxes))}
    gt_confusion = {g_idx:0 for g_idx in range(len(gt_bboxes))}
    pred_confusion = {d_idx:0 for d_idx in range(len(pred_bboxes))}

    # no_union means raw intersection, not intersection over union
    iou_matrix = _compute_ious(pred_bboxes, gt_bboxes, no_union=no_union)

    for d_idx in range(len(pred_bboxes)):
        for g_idx in range(len(gt_bboxes)):
            if not is_crowded(pred_bboxes[d_idx], gt_bboxes[g_idx], iou_matrix[d_idx, g_idx],
                              iou_thresh=iou_thresh, no_union=no_union):
                continue

            # IOU threshold met
            gt_crowding[g_idx] += 1
            pred_crowding[d_idx] += 1

            if pred_labels[d_idx] == gt_labels[g_idx]:
                gt_confusion[g_idx] += 1
                pred_confusion[d_idx] += 1

    return gt_crowding, pred_crowding, gt_confusion, pred_confusion



