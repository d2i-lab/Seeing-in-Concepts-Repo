import os
import json
import pycocotools
import pycocotools.mask
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import matplotlib

matplotlib.use('Agg')


class ImageHandler:
    def __init__(self, settings):
        self.seg_embeddings_dir = settings.seg_embeddings_dir
        self.coco_img_dir = settings.coco_img_dir
        self.sam_jsons_dir = settings.sam_jsons_dir

    def extract_segments(self, json_file):
        with open(json_file, 'r') as f:
            json_dict = json.load(f)
        segments = [seg['segmentation'] for seg in json_dict]
        decoded_segments = []
        for seg in segments:
            decoded_segments.append(pycocotools.mask.decode(seg))

        return decoded_segments

    def editImage(self, image_name: str, segment_id: int):
        img_path = os.path.join(self.coco_img_dir, image_name + '.jpg')
        seg_path = os.path.join(self.sam_jsons_dir, image_name + '.jpg.json')

        if not(os.path.isfile(img_path) and os.path.isfile(seg_path)):
            print(f"img_path: {img_path}", f"{os.path.isfile(img_path)}")
            print(f"seg_path: {seg_path}", f"{os.path.isfile(seg_path)}")
            return None
            

        img = plt.imread(img_path)

        segs = self.extract_segments(seg_path)
        segment_mask = np.ones_like(segs[0]) * -1
        segs = [seg for seg in segs if np.sum(seg) > 10]

        for i, seg in enumerate(segs): 
            segment_mask[np.nonzero(seg)] = i+1
        
        indices = np.where(segment_mask == segment_id)
        if len(indices[0]) == 0:
            print("skipping")


        min_x, min_y = min(indices[0]), min(indices[1])
        max_x, max_y = max(indices[0]), max(indices[1])

        imgcopy = img.copy()

        color_mask = np.zeros((imgcopy.shape[0], imgcopy.shape[1], 3), dtype=np.uint8)
        color_mask[indices] = [255, 0, 0]


        try:
            imgcopy = cv.addWeighted(imgcopy, 0.5, color_mask, 0.5, 0)
        except:
            pass
        cv.rectangle(imgcopy, (min_y, min_x), (max_y, max_x), (0,255, 0), 2)


        path = './temp.png'
        cv.imwrite(path, cv.cvtColor(imgcopy, cv.COLOR_RGB2BGR))

        return path
        