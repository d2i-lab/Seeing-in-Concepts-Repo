import unittest
import base64
import io
import os

import utils

import numpy as np
import h5py
from scipy.sparse import csr_matrix, vstack, load_npz, random


class BboxTests(unittest.TestCase):
    def setUp(self):
        img_size = (100, 100)
        mask = np.zeros(img_size)
        mask[0:10, 0:10] = 1 # Square of ones, top-left corner
        mask[90:100, 90:100] = 1 # Square, bottom-right corner
        self.mask = mask

    def test_subarray_1(self):
        p1 = (0,0)
        p2 = (10,10)
        subarr = utils.matrix_subarray(self.mask, [*p1, *p2], scale=0)
        self.assertEqual(subarr.flatten().sum(), 100)
        subarr2 = utils.matrix_subarray(self.mask, [*p1, *p2], scale=0.1)
        self.assertTrue(subarr.shape >= subarr2.shape)

    def test_subarray_2(self):
        p1 = (0,0)
        p2 = (100,100)
        subarr = utils.matrix_subarray(self.mask, [*p1, *p2], scale=0)
        self.assertEqual(subarr.flatten().sum(), 200)

    def test_union(self):
        '''
        Very basic test. Only exists to make sure union is not tampered with
        '''
        bbox1 = [10, 10, 50, 200]
        bbox2 = [0, 0, 100, 100]
        expected = [0, 0, 100, 200]
        self.assertEqual(utils.calculate_union(bbox1, bbox2), expected)

    def test_bbox_logic(self):
        '''
        Test to see if correct bounding box choices are made according to 
        different pairing scenarios:

        Cases:
        1. Both present: Bbox should be the union of the two (disputable)
        2. GT present only: FN. Should return None if include_FN=False
        3. DT present only: FP. Should return DT bbox only
        4. Both absent: Weird case. Shouldn't happen.
        '''

        # (GT, DT) pairs
        scenarios = [
            [[], []], # None
            [[1], [2]], # None
            [[], [0, 0, 10, 10]], # FP: No GT. Should be DT
            [[0, 0, 10, 10], []], # FN: No DT. Should be None
            [[0, 0, 10, 10], [10, 10, 15, 15]] # Both present: Should union
        ]
        expected = [
            None,
            None,
            'fp',
            'fn',
            'u'
        ]

        for (boxes, answer) in zip(scenarios, expected):
            with self.subTest(mode=boxes):
                result = utils.get_bbox(*boxes, False)
                if answer is None:
                    self.assertTrue(result is None)
                elif answer == 'fp':
                    self.assertTrue(result == boxes[1])
                elif answer == 'fn':
                    self.assertTrue(result is None)
                    result2 = utils.get_bbox(*boxes, True)
                    self.assertTrue(result2 == boxes[0])
                elif answer == 'u':
                    self.assertTrue(utils.calculate_union(*boxes) == result)


    def test_should_add_concept(self):
        bbox1 = [0, 0, 10, 10] # Full overlap. True.
        bbox2 = [8, 8, 30, 30] # 2x2 Overlap = Area 4. 4/200 < 0.1. False
        bbox3 = [8, 8, 10, 10] # 2x2 Overlap. Entire box is covered. True
        bbox4 = [6, 0, 30, 5] # (10-6)x(5)=20 Overlap. 20/200 == 0.1. True
        bbox5 = [20, 20, 30, 30] # No overlap
        thresh = 0.1
        self.assertTrue(utils.should_add_concept(self.mask, bbox1, thresh, 0))
        self.assertFalse(utils.should_add_concept(self.mask, bbox2, thresh, 0))
        self.assertTrue(utils.should_add_concept(self.mask, bbox3, thresh, 0))
        self.assertTrue(utils.should_add_concept(self.mask, bbox4, thresh, 0))
        self.assertFalse(utils.should_add_concept(self.mask, bbox5, thresh, 0))

class TestEncode(unittest.TestCase):
    def setUp(self):
        arrays = random(20, 1000, density=0.01, dtype=bool).A.astype(bool)
        sparse_arrays = [csr_matrix(x) for x in arrays]
        sparse_matrix = vstack(sparse_arrays)
        self.mat = sparse_matrix

    def test_sparse_to_npz(self):
        mat_npz = utils.sparse_to_npz(self.mat)
        npz_fake_file = io.BytesIO(mat_npz)
        mat = load_npz(npz_fake_file)
        self.assertTrue(np.array_equal(mat.todense(), self.mat.todense()))

    def test_save_to_hdf5(self):
        ious = np.random.normal(0.8, scale=1, size=self.mat.shape[0])
        config_str = '[fake config info]'
        labels = ['fake_labels_1','fake_labels_2']
        npz_str = utils.sparse_to_npz(self.mat)
        out_fname = 'delete.me.please'
        n = self.mat.shape[0]
        pred_bbox = ['[1,1,1,1]']
        gt_bbox = []
        data_pack = utils.DataPackage(
            img_path=['hi']*n,
            gt_label=[0]*n,
            pred_label=[0]*n,
            gt_bbox=gt_bbox,
            pred_bbox=pred_bbox,
            pred_score=[0.5]*n,
            fp_type1=[0]*n,
            fp_type2=[0]*n,
            fn=[0]*n,
            iou=ious,
        )

        def _decode_bin_list(lst:list)->list:
            return [x.decode('utf-8') for x in lst]

        utils.save_to_hdf5(out_fname, config_str, npz_str, labels, data_pack)
        with h5py.File(out_fname+'.hdf5', 'r') as f:
            decoding = base64.b64decode(f['sparse_matrix'][0])
            npz_fake_file = io.BytesIO(decoding)
            mat = load_npz(npz_fake_file)
            loaded_labels = [l.decode('utf-8') for l in f['labels']]
            loaded_img_paths = [l.decode('utf-8') for l in f['img_paths']]

            self.assertTrue(np.array_equal(mat.todense(), self.mat.todense()))
            self.assertTrue(f.attrs['config'] == config_str)
            self.assertTrue(loaded_labels == labels)

            # Begin testing rest of data...
            self.assertEqual(data_pack.img_path, loaded_img_paths)
            self.assertEqual(data_pack.gt_label, list(f['gt_labels']))
            self.assertEqual(data_pack.pred_label, list(f['pred_labels']))
            self.assertEqual(data_pack.gt_bbox, _decode_bin_list(list(f['gt_bboxes'])))
            self.assertEqual(data_pack.pred_bbox, _decode_bin_list(list(f['pred_bboxes'])))
            self.assertEqual(data_pack.pred_score, list(f['pred_scores']))
            self.assertEqual(data_pack.fp_type1, list(f['fp_type1s']))
            self.assertEqual(data_pack.fp_type2, list(f['fp_type2s']))
            self.assertEqual(data_pack.fn, list(f['fns']))
            self.assertEqual(data_pack.iou.tolist(), list(f['ious']))

        os.remove(out_fname+'.hdf5')
