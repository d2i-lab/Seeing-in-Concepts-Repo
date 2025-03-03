# Seeing in Concepts: Enabling Structured Image Representation and Analysis with Visual Concepts

This repository contains code for our paper, where we describe a new method for understanding images with visual concepts.
<figure>
<img src="/docs/overview.png" height="450">
<!-- <figcaption>A depiction of the VCR pipeline with image segmentation, concept formation, and data slicing</figcaption> -->
</figure>

*A depiction of the VCR pipeline with image segmentation, concept formation, and data slicing.*

Specifically this repository contains instructions on how to deploy our interface as well as instructions on replicating the evaluation.

## Pages
1. Interface Deployment (here) - for deploying the interface
2. [Custom Detection Results](/docs/custom-detections.md) - for creating custom detection results
3. [Evaluation](/docs/evaluation.md) - for setting up evaluation

## Datasets
For VCR, we use the MS-COCO dataset found [here](https://cocodataset.org). For each image dataset, we additionally require the concept data. 
To make deployment easier we have packaged COCO 2014 validation data [here (pw=coco-coco)](). Please download the data and unzip it accordingly. After extracting, run the following to download COCO 2014 validation directly from the website: `cd coco-dataset/imgs && bash download.sh`.

## Object Detection Results
While we provide COCO 2014 object detection result in the above dataset, users interested in evaluating their own detection results can create their own detection results by following our guide [here](/docs/custom-detections.md). If users want to simply visualize the demo, skip the linked guide.

## Interface Setup
The interface is implemented in two parts: (1) the frontend and (2) the backend. The frontend is written in Javascript with
the React framework, while the backend is implemented in Python with the FastAPI framework.

<figure>
<img src="/docs/ui-example-1.png" height="550">
<!-- <figcaption>Mining results from the interface.</figcaption> -->
</figure>

*Mining results from the interface.*

### Backend
> [!NOTE]
> Please make sure to satisfy the following Dependencies: (0) NVIDIA's cuML and Meta's Faiss Library (1) Dataset (2) Object Detection Results (3) Segment Extraction Results

The backend is responsible for concept-formation via clustering, mining concept results, and rendering concept visualizations.
Please make sure the dataset, object-detection results, and segment and embedding extraction results are prepared before reaching this point.

**Option 1: Install from NVIDIA and Meta Website (Recommended)**
* First, install NVIDIA's cuML library [here](https://docs.rapids.ai/install) (recommended). 
  ```bash
  # We found the following worked for us:
  conda install -n base conda-libmamba-solver
  conda create --solver=libmamba -n rapids-23.12 -c rapidsai -c conda-forge -c nvidia      rapids=23.12 python=3.10 cuda-version=11.8
  ```
* Next, install Meta's Faiss library:
  ```bash
  conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl
  ```
* Install dependencies from requirements.txt (make sure this pip is specific to the cuML+Faiss environment):
  ```bash
  miniconda3/envs/{env_name}/bin/python3 -m pip install -r requirements.txt
  ```
* Where "env_name" is the specific environment housing both Faiss and cuML

**Option 2: Install from Conda yml file:**
This method does not always work and we recommend following the typical steps above.
* Run `conda env create -f environment.yaml`

**After Option 1/2**
* Modify `ROOT_DIR` in [settings.py](/cluster-explorer/backend/api/utils/settings.py) to point to the dataset downloaded earlier. Modify `port` if needed as well.
* Specify a PORT number and run the backend: `PORT=3000 python3 server.py`

### Frontend
> [!NOTE]
> Please make sure to satisfy the following Dependencies: (0) NodeJS, React, and React-Bootstrap
* Install NodeJS and npm on your system. Follow a tutorial like [this one](https://www.digitalocean.com/community/tutorials/how-to-install-node-js-on-ubuntu-22-04) if needed.
* Ideally, start a new screen or tmux seshion to run the frontend code in the background (e.g. `tmux new -s "frontend"`)
* CD into the frontend `cd interface/frontend` 
* Run `npm install` to install required packages
* Update the [.env file](/cluster-explorer/frontend/.env) to point to the detection csv (e.g. `REACT_APP_BASE_CSV="coco_2014_val.csv"`) and backend api (if port or address changed)
* Start running the interface with `npm start`. This will start the UI on port 3001.
