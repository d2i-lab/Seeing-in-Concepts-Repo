from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse, FileResponse
from typing import List
import api.utils.settings as settings
import api.logic.cluster_handler as cluster_handler
import api.logic.image_handler as image_handler
import api.logic.export_handler as export_handler
import requests

router = APIRouter()
settings = settings.Settings()


cluster_handler = cluster_handler.ClusterHandler(settings)
image_handler = image_handler.ImageHandler(settings)
export_handler = export_handler.ExportHandler(settings)


@router.get('/generatecluster')
def cluster_endpoint(ncentroids: int, dimensionality: int, embeddingsType: str):
    result = cluster_handler.cluster_embeddings(ncentroids, dimensionality, embeddingsType)
    return JSONResponse(content=result)


# @router.post('/splitcluster')
# def split_endpoint(positive_examples: List[int], negative_examples: List[int], cluster_label: int):
#     result = cluster_handler.split_cluster(positive_examples, negative_examples, cluster_label)
#     return JSONResponse(content={"HELLO": "WORLD"})

@router.get('/images')
def image_endpoint(image_name: str, segment_id: int):
    img_path = image_handler.editImage(image_name, segment_id)
    if img_path is None: 
        raise HTTPException(status_code=404, detail='File not found')
    return FileResponse(img_path)

# TODO: Use pydantic models for request/response bodies
@router.post('/upload_remap_dict', status_code=status.HTTP_200_OK)
def upload_remap_dict(remap_dict: dict):
    res = export_handler.upload_remap_dict(remap_dict)
    if res == None:
        raise HTTPException(status_code=400, detail="File already exists")
    else:
        return {}

@router.post('/export', status_code=status.HTTP_200_OK)
async def export(export_dict: dict):
    res = export_handler.export(export_dict)
    if res == None:
        raise HTTPException(status_code=400, detail="File already exists")
    else:
        return JSONResponse(content=res)

@router.post('/import', status_code=status.HTTP_200_OK)
async def export(import_file: dict):
    res = export_handler.import_file(import_file)
    if res == None:
        raise HTTPException(status_code=400, detail="File does not Exist")
    else:
        return res

@router.post('/force-updatefile', status_code=status.HTTP_200_OK)
async def export(update_dict: dict):
    res = export_handler.update_file(update_dict, force=True)
    return res


@router.post('/updatefile', status_code=status.HTTP_200_OK)
async def export(update_dict: dict):
    res = export_handler.update_file(update_dict, force=False)
    return res

    
@router.get('/ping')
def ping():
    return {'ping': 'pong'}

@router.get('/')
def endpoint():
    return {"Hello": "saahir"}