
# CeNSE-DL

## Celestial Navigation in outer Space Enhanced by Deep Learning

This repository is contains code, paper and dataset generator tool used for term project for *Case Studies in ML* @ **UT Austin**.

## Steps to Reproduce

### Generate Dataset

**Dataset Generator application is modified implementation of [Starry Sky](https://github.com/Firnox/StarrySky) Unity Project by [Firnox](https://github.com/Firnox)**

Run `buildDataset.py` script provided in repository to generate dataset. Dataset can be generated fast in parellel fashion by setting number of workers which divide total number of image among themselves.

```python
python buildDataset.py {Total Number of Images} {Number of Workers}
```
### Customize your training environment and .ipynb

I am training on colab so, I uploaded dataset to google drive and connected drive to colab. But this step may be different for you.

**A better approach could have been to generate dataset on colab. If linux binary of DataGenerator is generated as it is, running the binary will produce empty image on colab as it is headless and certain modifications are needed to make it work on colab**

### Training Model

Run the notebook step by step to train model.
