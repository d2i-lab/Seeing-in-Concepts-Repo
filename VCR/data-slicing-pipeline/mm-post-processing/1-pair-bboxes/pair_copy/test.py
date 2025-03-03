import unittest

import numpy as np
import torch

from pair_confidence import PairConfidence
from pair_iou import PairIOU 
from crowd import check_crowding

class FakeBB:
    def __init__(self, bounds):
        self.tensor = torch.FloatTensor(bounds)

class DataItem:
    def __init__(self, path, gt_bboxes, gt_labels, pred_bboxes, pred_labels, pred_scores):
        self.img_path = path
        self.gt_bboxes = gt_bboxes
        self.gt_labels = gt_labels
        self.pred_bboxes = pred_bboxes
        self.pred_labels = pred_labels
        self.pred_scores = pred_scores
    
    def to_dict(self):
        d = {}
        d['img_path'] = self.img_path
        d['gt_bboxes'] = self.gt_bboxes
        d['gt_bboxes_labels'] = torch.tensor(self.gt_labels)

        d['pred_instances'] = {}
        d['pred_instances']['bboxes'] = torch.FloatTensor(self.pred_bboxes)
        d['pred_instances']['labels'] = torch.tensor(self.pred_labels)
        d['pred_instances']['scores'] = torch.tensor(self.pred_scores)
        return d

class TestConfidence(unittest.TestCase):
    def setUp(self):
        # (bbox coords, label)
        gt = [
            ([0,0,10,10], 0),
            ([10,10,20,20], 2),
            ([10,15,20,20], 9), # Should not match anything.
            ([15,15,20,20], 3),
        ]
        gt_bboxes = FakeBB([np.array(x[0]) for x in gt])
        gt_labels = [x[1] for x in gt]
        # (bbox coords, label, confidence)
        dt = [
            ([0,0,9,9], 0, 0.9), # Should match gt[0], label 0
            ([10,10,20,20], 2, 0.89), # Should match gt[1], label 2
            ([0,0,10,10], 1, 0.55), # Should not match anything. Only shared bbox is gt[0] (already matached)
            ([15,15,20,20], 3, 0.5), # Should match gt[3], label 3
            ([0,0,10,10], 10, 0.51), # Should not match anything. Only shared bbox is gt[0] (already matached)
            ([10,15,20,20], 9, 0.05), # Shoul not match anything. Conf too low
        ]
        path = 'Images/image.jpg'
        pred_bboxes = [pred[0] for pred in dt]
        pred_labels = [pred[1] for pred in dt]
        pred_scores = [pred[2] for pred in dt]

        images = [
            DataItem(path, gt_bboxes, gt_labels, pred_bboxes, pred_labels, 
                     pred_scores),
        ]
        data = []
        for image in images:
            data.append(image.to_dict())

        self.data = data

    def test_1(self):
        pairing = PairConfidence(self.data, 100, min_iou=0.1, 
                                 fp_iou_thresh=0.5, min_confidence=0.5,
                                 testing=True)
        df_pairs = pairing.pair()

        self.assertEqual(len(df_pairs), 6)
        self.assertEqual(df_pairs[df_pairs['gt_label'] == 0]['pred_label'].values[0], 0)
        self.assertEqual(df_pairs[df_pairs['gt_label'] == 2]['pred_label'].values[0], 2)
        self.assertEqual(df_pairs[df_pairs['gt_label'] == 9]['pred_label'].values[0], -1)

class TestIOU(unittest.TestCase):
    def setUp(self):
        # (bbox coords, label)
        gt = [
            ([0,0,10,10], 0),
            ([0,2,12,12], 1),
            ([10,10,15,15], 2),
            ([100,100,100,200], 9), # Should not match
            ([200,200,200,300], 10) # Should not match
        ]
        gt_bboxes = FakeBB([np.array(x[0]) for x in gt])
        gt_labels = [x[1] for x in gt]
        # (bbox coords, label, confidence)
        dt = [
            ([0,0,9,10], 0, 0.5), # Should match gt[0]
            ([0,0,8,10], 1, 0.9), # Should match gt[1]
            ([10,10,14,14], 2, 0.5), # Should match gt[2]
            ([10,10,15,19], 3, 0.5), # Should not match
        ]
        path = 'Images/image.jpg'
        pred_bboxes = [pred[0] for pred in dt]
        pred_labels = [pred[1] for pred in dt]
        pred_scores = [pred[2] for pred in dt]

        images = [
            DataItem(path, gt_bboxes, gt_labels, pred_bboxes, pred_labels, 
                     pred_scores),
        ]
        data = []
        for image in images:
            data.append(image.to_dict())

        self.data = data

    def test_1(self):
        pairing = PairIOU(self.data, min_iou=0.1, 
                                 fp_iou_thresh=0.5, min_confidence=0.5,
                                 testing=True)
        df_pairs = pairing.pair()
        self.assertEqual(df_pairs[df_pairs['gt_label'] == 0]['pred_label'].values[0], 0)
        self.assertEqual(df_pairs[df_pairs['gt_label'] == 1]['pred_label'].values[0], 1)
        self.assertEqual(df_pairs[df_pairs['gt_label'] == 2]['pred_label'].values[0], 2)
        self.assertEqual(df_pairs[df_pairs['gt_label'] == 9]['pred_label'].values[0], -1)
        self.assertEqual(df_pairs[df_pairs['gt_label'] == 10]['pred_label'].values[0], -1)

class TestCrowd(unittest.TestCase):
    def setUp(self):
        dt = [
            ([0,0,10,10], 0), # Intersects gt[1]
            ([5,5,10,10], 1), # Intersects gt[1]
            ([10,10,20,20], 2), # Intersects gt[2]
            ([11,15,20,20], 2) # Intersects gt[2]
        ]
        gt = [
            ([10,10,20,19], 2), # Intersects dt[2], dt[3]
            ([1,5,10,10], 1), # Intersects dt[0], dt[1]
            ([30,30,40,40], 3), # Intersects nothing
        ]

        self.pred_bboxes = [x[0] for x in dt]
        self.gt_bboxes = [x[0] for x in gt]
        self.pred_labels = [x[1] for x in dt]
        self.gt_labels = [x[1] for x in gt]
    
        print(self.pred_labels, self.gt_labels)

    def test_crowd(self):
        gt_crowding, pred_crowding, gt_confusion, pred_confusion = check_crowding(
            self.pred_bboxes, self.gt_bboxes, self.pred_labels, self.gt_labels, no_union=False)

        # Test gt_crowding
        self.assertEqual(gt_crowding[0], 2)
        self.assertEqual(gt_crowding[1], 2)
        self.assertEqual(gt_crowding[2], 0)

        # Test pred_crowding
        self.assertEqual(pred_crowding[0], 1)
        self.assertEqual(pred_crowding[1], 1)
        self.assertEqual(pred_crowding[2], 1)
        self.assertEqual(pred_crowding[3], 1)

        # Test gt_confusion
        self.assertEqual(gt_confusion[0], 2)
        self.assertEqual(gt_confusion[1], 1)
        self.assertEqual(gt_confusion[2], 0)

        # Test pred_confusion
        self.assertEqual(pred_confusion[0], 0)
        self.assertEqual(pred_confusion[1], 1)
        self.assertEqual(pred_confusion[2], 1)
        self.assertEqual(pred_confusion[3], 1)

class TestCrowdNoUnion(unittest.TestCase):
    def setUp(self):
        dt = [
            ([0,0,10,10], 0), # Intersects gt[1]
            ([0,0,200000,200000], 0)
        ]
        gt = [
            ([0,0,2000,2000], 0)
        ]
        self.pred_bboxes = [x[0] for x in dt]
        self.gt_bboxes = [x[0] for x in gt]
        self.pred_labels = [x[1] for x in dt]
        self.gt_labels = [x[1] for x in gt]

    def test1(self):
        gt_crowding, pred_crowding, gt_confusion, pred_confusion = check_crowding(
            self.pred_bboxes, self.gt_bboxes, self.pred_labels, self.gt_labels,
            no_union=True)

        self.assertEqual(gt_crowding[0], 2)
        self.assertEqual(pred_crowding[0], 1)
        self.assertEqual(pred_crowding[1], 1)


        gt_crowding, pred_crowding, gt_confusion, pred_confusion = check_crowding(
            self.pred_bboxes, self.gt_bboxes, self.pred_labels, self.gt_labels,
            no_union=False)

        self.assertEqual(gt_crowding[0], 0)
        self.assertEqual(pred_crowding[0], 0)
        self.assertEqual(pred_crowding[1], 0)


        