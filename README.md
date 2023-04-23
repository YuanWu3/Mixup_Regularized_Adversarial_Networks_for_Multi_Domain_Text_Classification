# Mixup_Regularized_Adversarial_Networks_for_Multi_Domain_Text_Classification
Implementation of "Mixup Regularized Adversarial Networks for Multi-Domain Text Classfication" in Pytorch (ICASSP 2021)

## Datasets
This folder contains the dataset in the same format as needed by our code. 

## Requirements:
- Python 3.6
- PyTorch 0.4
- PyTorchNet
- scipy
- tqdm (for progress bar)

## Training
All the parameters are set as the same as parameters mentioned in the article. You can use the following commands to the tasks:

## MDTC experiments on the Amazon review dataset

cd code/

```
python train.py --dataset prep-amazon --model mlp
```

## Citation
If you use this code for your research, consider citing:


    @inproceedings{wu2021mixup,
      title={Mixup regularized adversarial networks for multi-domain text classification},
      author={Wu, Yuan and Inkpen, Diana and El-Roby, Ahmed},
      booktitle={ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      pages={7733--7737},
      year={2021},
      organization={IEEE}
      }
