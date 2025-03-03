# Evaluation

We provide several bash scripts as well as the concept data for our datasets. Running the evaluation scripts requires several dependencies and it is recommended to setup different environments for both VCR and Domino.

## COCO 2014
To make deployment easier we have packaged COCO 2014 validation data [here (pw=coco-coco)](https://gtvault-my.sharepoint.com/:u:/g/personal/jxu680_gatech_edu/EQ2ujoG_siVDtQsPhig7p9wBqmH9x1M1wcyp25W0Yxvs0g?e=8gNwOP). Please download the data and unzip it accordingly. After extracting, run the following to download COCO 2014 validation directly from the website: `cd coco-dataset/imgs && bash download.sh`.

## Visual Genome (COCO Matching)
Similarly, we have the concept and segment data for Visual Genome part one. We shrink the dataset to include only images that have any of the original 80 coco classes. Please donwload the concept data [here (pw=coco-coco)](https://gtvault-my.sharepoint.com/:u:/g/personal/jxu680_gatech_edu/EUb5ta2C529MpPPZ6V0hb04B5H6HC6xQJJCGh-Soz8NVSA?e=MWCBBe) and run the download scripts as before.

## BDD Dataset
BDD Dataset is included
[here (pw=coco-coco)](https://gtvault-my.sharepoint.com/:u:/g/personal/jxu680_gatech_edu/EesUl3-nLaZNvk02tMOsmQgBVz96WtJ4iT95N2U6Chf4NQ?e=9i3bAe) but image download must be done via the BDD website. Users must register accounts and download from the site from there. Afterwards, users can run the bash script to extract relevant test images.

## VCR Scripts
The environment requirements for running VCR evaluation is the same as backend requirements (NVIDIA's CUML and FAISS) in the front-page README. Please follow those instructions [here](/README.md).

To run all relevant VCR evaluations, please configure the following three files to point to the correct files according to the downloaded datasets above:
* [coco tests](/slice-bench-tests-repo/curation/scripts/sweep/run_coco_vcr_sweep.sh)
* [vgg tests](/slice-bench-tests-repo/curation/scripts/sweep/run_vgg_vcr_sweep.sh)
* [bdd tests](/slice-bench-tests-repo/curation/scripts/sweep/run_bdd_vcr_sweep.sh)

Then simply run the bash scripts according to the dataset (e.g. `bash run_coco_vcr_sweep.sh`)

## DOMINO Scripts

### DOMINO Dependencies
We first need to download DOMINO's dependencies.

**Option 1**

Install Domino following directions from [the repository itself](https://github.com/HazyResearch/domino). We found Domino==0.1.5 to work the best for us.

**Option 2**

Install Domino from our [test-domino.yml](/test-domino.yml). This is known to have issues, but can be used as a reference.

### Testing
Like VCR, after all dependencies are met, we can simply run the bash scripts in the `scripts/sweep` folder:
* [coco tests](/slice-bench-tests-repo/curation/scripts/sweep/run_coco_domino_sweep.sh)
* [vgg tests](/slice-bench-tests-repo/curation/scripts/sweep/run_vgg_domino_sweep.sh)
* [bdd tests](/slice-bench-tests-repo/curation/scripts/sweep/run_bdd_vcr_sweep.sh)

## Viewing Evaluation Results
Once all results are run, the graph can be viewed by running the [compare notebook.](/slice-bench-tests-repo/curation/compare_new.ipynb)!
