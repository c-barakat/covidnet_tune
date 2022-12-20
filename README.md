# Applying Hyperparameter Tuning to COVID-Net
This repository includes the code and data required to replicate the results of the manuscript "Analysis of Chest X-Ray as a Use Case for an HPC-enabled Data Analysis and Machine Learning Platform for Medical Diagnosis Support."

Please note that some of the code is streamlined for parallel execution and will need to be adapted for serial implementation.

# Datasets
* The COVIDx dataset can be obtained by following the process described in the [original COVID-Net repository](https://github.com/lindawangg/COVID-Net/blob/master/docs/COVIDx.md).
* The E*HealthLine (EHL) dataset can be obtained from the [FZ-Juelich B2Share website](https://b2share.fz-juelich.de/records/aef5d3b8aa044485b9620b95b60c47a2).

The dataset train/test split was done as part of a [previous publication](https://doi.org/10.23919/MIPRO55190.2022.9803320), and can be replicated based on the train/test split described in the current manuscript and the train_fusion_0.8.txt and test_ehl_edited_0.8.txt files.

# Model
* COVID-Net can be obtained from the [COVID-Net repository](https://github.com/lindawangg/COVID-Net)
* Due to storage restrictions, this repository will not include the trained models based on each of the best parameters provided by the 4 selected Ray Tune schedulers. Instead, [training_parameters.csv](training_parameters.csv) contains the parameters produced by each of the schedulers, and can be used to train the downloaded COVID-Net model.

# Software, Packages, and versions
## Required for parallel execution
* Tensorflow-GPU==1.13.1
* Horovod==0.16.2
* cuDNN==7.5.1.10
* CUDA==10.1.105
* ParaStationMPI==5.4.0-1
* NCCL==2.4.6-1
* mpi4py==3.1.4

## Required for serial execution
* Python==3.6.8
* Tensorflow==1.13.1
* matplotlib==3.0.3
* numpy==1.19.0
* opencv-python==4.4.0.44
* pandas==0.24.2
* ray==0.6.2
* simplejson==3.16.0

# Repository Structure
* [covidnet_outputs](covidnet_outputs/) directory should be created to contain the saved models after training, as well as the pretrained COVID-Net model.
* [data](data/) is subdivided into train and test subdirectories containing the images from the downloaded datasets.
* [analyse_results.ipynb](analyse_results.ipynb) will export graphs to the [figures](figures/) directory.
* [results](results/) contains the prediction outputs from running [testing.py](testing.py).
* Further directories can or will need to be created when [Ray Tune is initialised](covidnet_on_fusion.py) or when [training_hvd.py](training_hvd.py) is run.
