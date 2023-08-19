# X-MoRe: Cross-Modal Retrieval Meets Inference: Improving Zero-Shot Classification with Cross-Modal Retrieval

## Overview
This repository contains the implementation of X-MoRe : a novel inference based zero-shot method using cross-modal retrieval.

## Prerequisites

### Hardware

This implementation is for the single-GPU configuration. This codebase is tested on a GPU with 24GB memory.


### Environment 
The code is tested on PyTorch 1.13.1. 

### Datasets 

We suggest downloading all datasets to a root directory (`${data_root}`), and renaming the directory of each dataset as suggested in `${ID_to_DIRNAME}` in `./data/datautils.py`. This would allow you to evaluate multiple datasets within the same run.     
If this is not feasible, you could evaluate different datasets separately, and change the `${data_root}` accordingly in the bash script.

For out-of-distribution generalization, we consider 5 datasets:

* [ImageNet](https://image-net.org/index.php) 
* [ImageNet-A](https://github.com/hendrycks/natural-adv-examples)
* [ImageNet-R](https://github.com/hendrycks/imagenet-r)
* [ImageNet-V2](https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-matched-frequency.tar.gz)
* [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch)

For cross-datasets generalization, we consider 10 datasets:
* [Flower102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz)
* [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz)
* [OxfordPets](https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz)
* [StanfordCars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)
* [UCF101](https://drive.google.com/file/d/10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O/view?usp=sharing)
* [Caltech101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz)
* [Food101](http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz)
* [SUN397](http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz)
* [Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz)
* [EuroSAT](http://madm.dfki.de/files/sentinel/EuroSAT.zip)

For cross-dataset generalization, we adopt the same train/val/test splits as CoOp. Please refer to [this page](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md#how-to-install-datasets), and look for download links of `split_zhou_${dataset_name}.json`, and put the json files under `./data/data_splits/`. 


## Run X-MoRe

We provide bash scripts under `./scripts`. You can modify the paths and other args in the scripts.     

An example to reproduce 10 image classification tasks in Table1, you can run

```
bash ./scripts/test_FS_RN50.sh gpu_num 
bash ./scripts/test_FS_VIT.sh gpu_num
```
or domain generalization task in Table 2 with
```
bash ./scripts/test_OOD_RN50.sh gpu_num 
bash ./scripts/test_OOD_VIT.sh gpu_num
```

The command line arg `${testsets}` can be multiple test datasets split by "/" (, which are stored under the same root dir `${data_root}`).    
Note that for simplicity, we use `set_id` to denote different datasets. A complete list of `set_id` can be found in `${ID_to_DIRNAME}` in `./data/datautils.py`. 
