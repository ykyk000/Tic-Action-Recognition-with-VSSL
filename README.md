
Abstract:
Tic Disorder (TD) is a neurological disorder in children accompanied by repetitive, involuntary facial or body tics and uncontrollable vocal tics. Early diagnosis and proper medical therapy of tic disorder can improve children’s condition. In this paper, we aim to recognize tic actions caused by the tic disorder from videos. We have collected a tic dataset containing videos from 80 children with tic disorder. The dataset is annotated as 13 categories of motor tic actions. We study the consistency-based video semi-supervised learning method which explores the spatio-temporal consistency characters for tic action recognition, in such case we have some labeled and unlabeled tic action videos. Specifically, we use a backbone network to predict both action class labels and action location maps. Both the labeled and unlabeled videos are augmented and input into the network. The output of original videos and its augmentation view are enforced to be consistent with each other. For the classification branch, we utilize distribution consistency to minimize the distribution distance between the learned features from original and augmentation view. For the localization branch, we propose the motion pattern consistency, which constrains different video views to have similar motion patterns. We also design the multi-resolution feature fusion modules in backbone networks to further improve the action representation capability. We compare the proposed approach with other supervised and semi-supervised action recognition methods on tic action dataset. To give a fair comparison, we also conduct experiments on some public action recognition datasets, UCF101-24 and JHMDB-21. The experimental results demonstrate the effectiveness of our video semi-supervised approach for tic action recognition.

1.Tic dataset cannot be disclosed temporarily due to its involvement in patient privacy

2.After organizing the remaining parts, they will be released together
Link to download I3D pre-trained weights:  
```
https://github.com/piergiaj/pytorch-i3d/tree/master/models
```
We have used **rgb_charades.pt** for our experiments.

## Datasets Info

UCF101-24 splits: [Pickle files](https://drive.google.com/drive/u/0/folders/1aFlPKtzWIufyAOkcAmUySH4PB_uCPDkj)

JHMDB-21  splits: [Text files](https://drive.google.com/drive/u/0/folders/1whGR2pg299D5W7jDV9Rop_jpr1ENIALF)


