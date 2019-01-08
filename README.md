The code of this repository was used for the [following publication](https://www.sciencedirect.com/science/article/pii/S0031320318303091). If
you find this code useful please cite our paper:

```
@article{Gallego2017,
title = "A selectional auto-encoder approach for document image binarization",
author = "Jorge Calvo-Zaragoza and Antonio-Javier Gallego",
journal = "Pattern Recognition",
year = "2018"
}
```

Below we include instructions to binarize and reproduce the experiments.



## Binarize

The `binarize.py` script performs the binarization of an input image using a trained model. The parameters of this script are the following:


| Parameter    | Default | Description                      |
| ------------ | ------- | -------------------------------- |
| `-imgpath`   |         | Path to the image to process     |
| `-modelpath` |         | Path to the model to load        |
| `-w`         |  256    | Input window size                |
| `-s`         |  -1     | Step size. -1 to use window size |
| `-f`         |  64     | Number of filters                |
| `-k`         |  5      | Kernel size                      |
| `-drop`      |  0      | Dropout percentage               |
| `-stride`    |  2      | Convolution stride size          |
| `-every`     |  1      | Residual connections every x layers |
| `-th`        |  0.5    | Selectional threshold            |
| `-save`      |         | Output image filename            |
| `--demo`     |         | Activate demo mode               |


The mandatory parameters are `-imgpath` and `-modelpath`, the rest are optional. You also have to choose if you want to see a demo (`--demo`) or to save (`-save`) the binarized image.

For example, to binarize the image `img01.png` using the model trained for Dibco 2016 and the parameters specified in the paper, you have to run the following command:


```
$ python binarize.py -imgpath img01.png -modelpath MODELS/model_weights_dibco_6_256x256_s96_aug_m205_f64_k5_s2_se3_e200_b32_esp.h5 -w 256 -s 96 -f 64 -k 5 -stride 2 -th 0.5 --demo
```

The `MODELS` folder includes the trained models for the datasets evaluated.



## Train

The `train.py` script performs the training of the proposed network from a dataset of images and a series of input parameters that allow to configure both the training process and the network topology. This executable, in addition to training, performs the validation and saves the resulting images.

The parameters of this script are the following:


| Parameter    | Default | Description                      |
| ------------ | ------- | -------------------------------- |
| `-path`      |         | Base path to datasets            |
| `-db`        |         | Database name ['dibco','palm','phi','ein','sal','voy','bdi','all']    |
| `-dbp`       |         | Database dependent parameters [dibco fold, palm gt]     |
| `--aug`      |         | Load augmentation folders        |
| `-w`         |  256    | Input window size                |
| `-s`         |  -1     | Step size. -1 to use window size |
| `-f`         |  64     | Number of filters                |
| `-k`         |  5      | Kernel size                      |
| `-drop`      |  0      | Dropout percentage               |
| `-stride`    |  2      | Convolution stride size          |
| `-every`     |  1      | Residual connections every x layers |
| `-th`        |  0.5    | Selectional threshold. -1 to evaluate from 0 to 1  |
| `-page`      |  -1     | Page size to divide the training set. -1 to load all |
| `-start_from`|  0      | Start from this page             |
| `-super`     |  1      | Number of super epochs           |
| `-e`         |  200    | Number of epochs                 |
| `-b`         |  10     | Mini-batch size                  |
| `-esmode`    |  p      | Early stopping mode: g='global', p='per page' |
| `-espat`     |  10     | Early stopping patience          |
| `-verbose`   |  1      | 1=show batch increment, other=mute |
| `--test`     |         | Only run test (deactivate training) |
| `--show`     |         | Show the output images           |
| `-loadmodel` |         | Weights filename to load for test or for initialization  |


The only mandatory parameters are `-path` and `-db`. To run the training you must first indicate the path where the datasets are located (with the `-path` parameter) and the name of the dataset to evaluate (with the `-db` parameter). Depending on the name of the dataset the system will create the partitions for training and validation.

The folders of each dataset must have a specific name (see table in section "Datasets" or consult the source code). Within each folder there must be two subfolders, one with the suffix `\_GR` with the images in gray scale, and another with the suffix `\_GT` for the ground truth.

Option `--aug` activates the data augmentation, in this case the system will load the images stored in the folder with prefix `aug_`.

For example, to train a network for the dataset Dibco 2016, using data augmentation and the parameters specified in the paper, you have to run the following command:


```
$ python -u train.py -path datasets -db dibco -dbp 6 --aug -w 256 -s 128 -f 64 -k 5 -e 200 -b 10 -th -1 -stride 2 -page 64
```


Once the script finishes, it will save the learned weights and the binarized images resulting from the validation partition in a folder with the prefix `_PR-` followed by the name of the model. It will also show the mean squared reconstruction error and the obtained F-m. We have also used the evaluation tool provided by DIBCO (http://vc.ee.duth.gr/h-dibco2016/benchmark/), which allowed us to validate the obtained F-m and also calculate the pseudo F-m.



## Datasets

Below is a summary table of the datasets used in this work along with a link from which they can be downloaded:


| Dataset      | Acronym | URL     |
| ------------ | ------- | ------- |
| DIBCO 2009   | DB09    | http://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/   |
| DIBCO 2010   | DB10    | http://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/ |
| DIBCO 2011   | DB11    | http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/        |
| DIBCO 2012   | DB12    | http://utopia.duth.gr/~ipratika/HDIBCO2012/benchmark/       |
| DIBCO 2013   | DB13    | http://utopia.duth.gr/~ipratika/DIBCO2013/benchmark/        |
| DIBCO 2014   | DB14    | http://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/  |
| DIBCO 2016   | DB16    | http://vc.ee.duth.gr/h-dibco2016/benchmark/                 |
| Palm Leaf    | PL-I / PL-II | http://amadi.univ-lr.fr/ICFHR2016_Contest/             |
| Persian docs | PHI     | http://www.iapr-tc11.org/mediawiki/index.php/Persian_Heritage_Image_Binarization_Dataset_(PHIBD_2012) |
| Ensiedeln    | ES      | http://www.e-codices.unifr.ch/en/sbe/0611/                  |
| Salzinnes    | SAM     | https://cantus.simssa.ca/manuscript/133/                    |


