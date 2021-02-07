EchoNet-Labs:<br/>Deep Learning Prediction of Biomarkers from Echocardiogram Videos
------------------------------------------------------------------------------

![A training dataset of over seventy thousand echocardiogram videos and paired biomarker values from the same patient were used to train a video-based AI system for prediction of laboratory values. Our deep learning based AI system used spatio-temporal convolutions to infer biomarker values from both anatomic (spatial) and physiologic (temporal) information contained with echocardiogram videos. To understand the relative importance of spatial and temporal information, ablation datasets removing texture, motion, and extracardiac structures were adopted to perform interpretations experiments. ](media/figure_1.png)

EchoNet-Labs is an end-to-end deep learning model for predicting 14 different biomarkers and lab values from echocardiogram videos. 

For more information, check out our pre-print:
> [**Deep Learning Prediction of Biomarkers from Echocardiogram Videos**](https://www.medrxiv.org/content/10.1101/2021.02.03.21251080v1)
  J. Weston Hughes, Neal Yuan, Bryan He, Jiahong Ouyang, Joseph Ebinger, Patrick Botting, Jasper Lee, James E. Tooley, Koen Neiman, Matthew P. Lungren, David Liang, Ingela Schnittger, Robert A. Harrington, Jonathan H. Chen, Euan Ashley, Susan Cheng, David Ouyang, James Zou. 2021.

Dataset
-------
We plan to share corresponding lab values to the 10,030 echocardiogram images which were released with [EchoNet-Dynamic](https://echonet.github.io/dynamic/).

Model Performance
-------
EchoNet-Labs performs well predicting a range of lab values both on data from the medical system where it was trained and other medical centers:
![Scatterplots (top) and receiver-operating characteristic (ROC) curves (bottom) for prediction of (A) hemoglobin, (B) B-Type Natriuretic Peptide, (C) Blood Urea Nitrogen, and (D) Troponin I. Blue points and curves denote to a held-out test set of patients from Stanford Medicine not previously seen during model training. Red points and curves denote to performance on the external test set from Cedars-Sinai Medical Center. Black curves denote a benchmark with linear regression using demographics and echocardiogram features (LVEF, RVSP, Heart Rate) on the Stanford test set.
](media/figure_2.png)

Installation
------------

First, clone this repository and enter the directory by running:

    git clone <this repo link>
    cd labs

EchoNet-Labs is implemented for Python 3, and depends on the following packages:
  - NumPy
  - PyTorch
  - Torchvision
  - OpenCV
  - skimage
  - sklearn
  - tqdm

Echonet-Labs and its dependencies can be installed by navigating to the cloned directory and running

    pip install --user .

Usage
-----
### Preprocessing DICOM Videos

The input of EchoNet-Labs is an apical-4-chamber view echocardiogram video of any length. The easiest way to run our code is to use videos from our dataset, but we also provide in [EchoNet-Dynamic](https://github.com/echonet/dynamic/blob/master/scripts/ConvertDICOMToAVI.ipynb) a notebook `ConvertDICOMToAVI.ipynb`, to convert DICOM files to AVI files used for input to EchoNet-Dynamic and EchoNet-Labs. The Notebook deidentifies the video by cropping out information outside of the ultrasound sector, resizes the input video, and saves the video in AVI format. 

### Setting Path to Data

By default, EchoNet-Dynamic assumes that a copy of the data is saved in a folder named `a4c-video-dir/` in this directory.
This path can be changed by creating a configuration file named `echonet.cfg` (an example configuration file is `example.cfg`). This path can also be overwritten as an argument to `echonet.utils.video.run`.

### Running Code
Echonet-Labs trains models to predict lab values based on both full video data and ablated input data, to better understand which features are necessary to make predictions

#### Prediction of a lab value from Subsampled Clips

    cmd="import echonet; echonet.utils.video.run(modelname=\"r2plus1d_18\",
                                                 tasks=\"logBNP\",
                                                 frames=32,
                                                 period=2,
                                                 pretrained=True,
                                                 batch_size=8)"
    python3 -c "${cmd}"

This creates a directory in `output/video`, which will contain
  - `log.csv`: training and validation losses
  - `best.pt`: checkpoint of weights for the model with the lowest validation loss
  - `valid_predictions.csv`: estimates of logBNP on the validation set. Running again setting `test=True` will produce test_predictions.csv
  
#### Prediction of a lab value from Ablated Clips

Setting `segmentation_mode="only"` trains and validates a model solely on segmentations produced from EchoNet-dynamic (segmentations need to be pre-generated). Setting `segmentation_mode="both"` trains and validates a model with only the left ventricle visible. Setting `single_repeated=True` trains and video model on a single frame of input.

### Citations
> [**Deep Learning Prediction of Biomarkers from Echocardiogram Videos**](https://www.medrxiv.org/content/10.1101/2021.02.03.21251080v1)
  J. Weston Hughes, Neal Yuan, Bryan He, Jiahong Ouyang, Joseph Ebinger, Patrick Botting, Jasper Lee, James E. Tooley, Koen Neiman, Matthew P. Lungren, David Liang, Ingela Schnittger, Robert A. Harrington, Jonathan H. Chen, Euan Ashley, Susan Cheng, David Ouyang, James Zou. 2021.

> [**Video-based AI for beat-to-beat assessment of cardiac function**](https://www.nature.com/articles/s41586-020-2145-8)<br/>
  David Ouyang, Bryan He, Amirata Ghorbani, Neal Yuan, Joseph Ebinger, Curt P. Langlotz, Paul A. Heidenreich, Robert A. Harrington, David H. Liang, Euan A. Ashley, and James Y. Zou. <b>Nature</b>, March 25, 2020.
  
  
