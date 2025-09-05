# Interactive Triplet Attention for Few-Shot Fine-Grained Image Classification

## Code environment
This code requires Pytorch 1.7.0 and torchvision 0.8.0 or higher with cuda support. It has been tested on Ubuntu 16.04. 

You can create a conda environment with the correct dependencies using the following command lines:
```
conda env create -f environment.yml
conda activate ITAM
```
## Train and test
For fine-grained few-shot classification, we provide the training and inference code for ITAM, as they appear in the paper. 

To train a model from scratch, simply navigate to the appropriate dataset/model subfolder in `experiments/`. Each folder contains 3 files: `train.py`, `train.sh` and `test.py`. Running the shell script `train.sh` will train and evaluate the model with hyperparameters matching our paper. Explanations for these hyperparameters can be found in `trainers/trainer.py`.

For example, to train ITAM on `CUB_fewshot_cropped` with ResNet-12 as the network backbone under the 1-shot setting, run the following command lines:
```
cd experiments/CUB_fewshot_cropped/ITAM/ResNet-12
./train.sh
```







