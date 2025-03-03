import os

from pydantic_settings import BaseSettings

ROOT_DIR = '/data/coco-dataset'

class Settings(BaseSettings):
    port: int = 4443
    data_dir: str = os.path.join(ROOT_DIR, 'saved')
    label_dir: str = os.path.join(ROOT_DIR, 'class_labels.txt')
    img_dir: str = '../temp-imgs'
    coco_img_dir: str = os.path.join(ROOT_DIR, 'imgs', 'val2014')
    sam_jsons_dir: str = os.path.join(ROOT_DIR, 'segments')
    seg_embeddings_dir: str = os.path.join(ROOT_DIR, 'embeds', 'clip')

    # Pickled data for faster loading
    fast_path_dir: str = os.path.join(ROOT_DIR, 'fast_pkls', 'coco-2014-val-clip-embeds-fast.pkl')
    pretrained_clip_labels_dir: str = os.path.join(ROOT_DIR, 'fast_pkls', 'ViT16_clip_text.pkl')
    umap_embeddings_dir: str = os.path.join(ROOT_DIR, 'coco-2014-val-clip-embeds-umap.pkl')

    # Miner cache parameters
    max_files: int = 2
    img_cache_size: int = 10
    result_cache_size: int = 10

    # Mining parameters
    limit_num_rows: int = 100_000