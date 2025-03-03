import os
import numpy as np
import pandas as pd
import concurrent
from scipy.sparse import csr_matrix
import json
import h5py

import api.utils.utils as utils
import api.utils.data_utils as data_utils

class ExportHandler:
    def __init__(self, settings):
        self.WORKERS = 10
        self.ROWS_PER_WORKER = 5000
        self.remap_output_dir = settings.data_dir
    
    def import_file(self, import_file):
        fname = import_file['file']
        outpath = os.path.join(self.remap_output_dir, fname)
        if not os.path.exists(outpath):
            print("This should never happen")
            return None

        labels = []
        with h5py.File(outpath, 'r') as f:
            labels = [l.decode('utf-8') for l in list(f['labels'])]
        
        return {"labels": labels}
        

    def export(self, export_dict):
        '''
        Convert the segment bbox csv file to a concept HDF5 file.
        '''
        # TODO: Why is the output fname stored in metadata as opposed to a param?
        print(export_dict.keys())
        export_dict = export_dict['exportDict']
        fname = export_dict['metadata']['fname']
        outpath = os.path.join(self.remap_output_dir, fname)
        if os.path.exists(outpath):
            return None

        segment_bbox_path = export_dict['metadata']['segment_bbox_path']

        # concept_mapping = export_dict['cluster_map']
        # [img_name][seg_id] = concept_id
        img_segmap = utils.get_segment_concept_per_image(export_dict)
        df = pd.read_csv(segment_bbox_path)

        interacting_as_strings = df['interacting_segments'].tolist()

        df['interacting_segments'] = df['interacting_segments'].map(
            lambda x: utils.parse_bbox_str(x, dtype=int))

        n_concepts = len(export_dict['cluster_map'])

        # Set the values of the matrix by remapping the segment ids to concept ids
        print('Starting workers')
        with pd.read_csv(segment_bbox_path, chunksize=self.ROWS_PER_WORKER) as reader:
            # with Pool(WORKERS) as pool:
            #     args = [(chunk, n_concepts, img_segmap) for chunk in reader]
            #     results = pool.map(worker_fn, args)
            with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
                results = executor.map(self.worker_fn, [(chunk, n_concepts, img_segmap) for chunk in reader])
        print('workers done')

        # mat = np.zeros((len(df), n_concepts), dtype=bool)
        mat = np.zeros((len(df), n_concepts), dtype=np.uint8)
        for result in results:
            for idx, row in result:
                mat[idx] = row

        # df_csv_out = pd.DataFrame(mat, columns=[str(i) for i in range(n_concepts)])
        # df_csv_out.to_csv('{}.csv'.format('debug.csv'), index=False)

        # Save the matrix to an HDF5 file
        package = data_utils.DataPackage(
            img_path = df['img_path'].tolist(),
            gt_label = df['gt_label'].tolist(),
            pred_label = df['pred_label'].tolist(),
            gt_bbox = df['gt_bbox'].tolist(),
            pred_bbox=df['pred_bbox'].tolist(),
            ignore_flag=df['ignore_flag'].tolist(),
            pred_score=df['pred_score'].tolist(),
            fp_type1=df['fp_type1'].tolist(),
            fp_type2=df['fp_type2'].tolist(),
            fn=df['fn'].tolist(),
            iou=df['iou'].tolist(),
            gt_crowd=df['gt_crowding'].tolist(),
            pred_crowd=df['pred_crowding'].tolist(),
            gt_confusion=df['gt_confusion'].tolist(),
            pred_confusion=df['pred_confusion'].tolist(),
        )


        labels = []
        for concept_id in range(n_concepts):
            labels.append(export_dict['cluster_names'][str(concept_id)])

        print(labels)
        # sparse_mat = csr_matrix(mat, dtype=bool)
        sparse_mat = csr_matrix(mat, dtype=np.uint8)
        npz_mat = data_utils.sparse_to_npz(sparse_mat)
        data_utils.save_to_hdf5('{}.h5'.format(outpath), '', npz_mat, labels, 
                            package, concept_mapping=img_segmap, interacting=interacting_as_strings)

        return {}

    def force_update(self, update_dict):
        pass


    def capture_last_number(self, new_label, label):
        import re
        pattern = re.escape(new_label) + r'_([^_]+)$'
        # print("searching for", new_label, label, pattern)
        match = re.search(pattern, label)
        if match:
            return match.group(1)
        else:
            return None

    def update_file(self, update_dict, force=False):
        fname = update_dict['file']
        data = update_dict['data'] # data maps [old label] to new label

        if 'pairing' in update_dict:
            data = {k: v for k, v in update_dict['pairing'].items()}
            # data = {}
            print('these are my pairs')
            print(data)


        # print(data)
        outpath = os.path.join(self.remap_output_dir, fname)
        if not os.path.exists(outpath):
            print("This should never happen")
            return 'error'

        # Check if labels overlap
        labels = []
        with h5py.File(outpath, 'r') as f:
            labels = [l.decode('utf-8') for l in list(f['labels'])]

        if not force:
            conflict = False
            conflict_labels = []
            for old_label in data.keys():
                new_label = data[old_label]
                if new_label == old_label: 
                    continue
                if new_label in set(labels):
                    conflict = True
                    conflict_label = new_label # Just being explicit
                    conflict_labels.append((old_label, conflict_label))

            # Find suggested words
            suggested_labels = []
            if conflict:
                max_val = 0
                for (old_label, conflict_label) in conflict_labels:
                    if "_" not in conflict_label: base_label = conflict_label
                    else: base_label, _ = conflict_label.rsplit('_', 1)
                    for label in set(labels):
                        end_number = self.capture_last_number(base_label, label)
                        if (end_number is not None) and (end_number.isdigit()):
                            max_val = max(max_val, int(end_number))

                    new_suggestion = base_label + '_' + str(max_val + 1)
                    # pair = [conflict_label, new_suggestion]
                    pair = (old_label, new_suggestion)
                    # suggested_labels.append(base_label + '_' + str(max_val + 1))
                    suggested_labels.append(pair)
                return {'conflict': suggested_labels, 'success': False}

        # NO CONFLICTS case
        # OR FORCE

        for i, label in enumerate(labels):
            if label in data:
                labels[i] = data[label]
        
        with h5py.File(outpath, 'r+') as f:
            f['labels'][...] = labels 

        return {'confict': [], 'success': True}
        


    def upload_remap_dict(self, remap_dict: dict):
        fname = remap_dict['metadata']['fname']
        outpath = os.path.join(self.remap_output_dir, fname)
        if os.path.exists(outpath):
            return None

        with open(outpath, "w") as f:
            json.dump(remap_dict, f)

        return {}

    def worker_fn(self, args):
        '''
        Single worker function for parallel processing of the segment bbox csv file.
        Specifically, this function remaps the segment ids to concept ids.
        '''
        (df_chunk, n_concepts, img_segmap) = args
        # concept_row = np.zeros((len(df_chunk), n_concepts), dtype=bool)

        # uint8 for counting--beware of overflow!
        concept_row = np.zeros((len(df_chunk), n_concepts), dtype=np.uint8)
        index_arr = np.array(df_chunk.index)

        start = index_arr[0]
        df_chunk['interacting_segments'] = df_chunk['interacting_segments'].map(
            lambda x: utils.parse_bbox_str(x, dtype=int))

        for i, row in df_chunk.iterrows():
            img_name = row['img_path'].split('.')[0]
            interacting_segments = row['interacting_segments']
            # interacting_concepts = []
            interacting_concepts_counter = {}
            if img_name not in img_segmap:
                print('could not find', img_name)
                continue

            mapping = img_segmap[img_name]
            if len(interacting_segments) == 0:
                continue

            found_segments = []
            for seg_id in interacting_segments:
                seg_id = seg_id + 1
                if seg_id not in mapping:
                    continue

                found_segments.append(seg_id)
                concept_id = mapping[seg_id]
                interacting_concepts_counter[concept_id] = interacting_concepts_counter.get(concept_id, 0) + 1
                # interacting_concepts.append(concept_id)

            if len(found_segments) == 0:
                continue
                
            # concept_row[i-start, interacting_concepts] = True
            cids = [k for k, v in interacting_concepts_counter.items() if v >= 1]
            counts = [interacting_concepts_counter[k] for k in cids]
            concept_row[i-start, cids] = counts
            # concept_row[i-start, interacting_concepts] = True
        
        # list: (idx: int, row: np.array)
        return [(idx, concept_row[i]) for i, idx in enumerate(index_arr)]