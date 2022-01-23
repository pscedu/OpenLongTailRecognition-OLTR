# Large-Scale Long-Tailed Recognition in an Open World

* Refer to the master branch for latest code 

Further information please contact [Rajanie Prabha](mailto:rajanie.prabha@gmail.com).

## Requirements
* [PyTorch](https://pytorch.org/) (version >= 0.4.1)
* [scikit-learn](https://scikit-learn.org/stable/)

## Data Preparation
For the dataset, visit [MiikeMineStamps](https://kukuruza.github.io/MiikeMineStamps/).


### Stamps Dataset
- Stage 1 training:
```
python main_shuffler.py --config config/stamps/stage_1.py
```
- Stage 2 training (At this stage, only single-GPU is supported, please switch back to single-GPU training.):
```
python main_shuffler.py --config config/stamps/stage_2_meta_embedding.py
```
- Close-set testing:
```
python main_shuffler.py --config config/stamps/stage_2_meta_embedding.py --test
```
- Open-set testing (thresholding)
```
python main_shuffler.py --config config/stamps/stage_2_meta_embedding.py --test_open
```

## Things to Note:
1. Please keep number of classes as total number of classes + 1.

## License and Citation
```
@article{Buitrago2021,
author = "Paola Buitrago and Evgeny Toropov and Rajanie Prabha",
title = "{MiikeMineStamps Dataset}",
year = "2021",
month = "5",
url = "https://kilthub.cmu.edu/articles/dataset/MiikeMineStamps_Dataset/14604768",
doi = "10.1184/R1/14604768.v1"
}
```
```
@inproceedings{openlongtailrecognition,
  title={Large-Scale Long-Tailed Recognition in an Open World},
  author={Liu, Ziwei and Miao, Zhongqi and Zhan, Xiaohang and Wang, Jiayun and Gong, Boqing and Yu, Stella X.},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
