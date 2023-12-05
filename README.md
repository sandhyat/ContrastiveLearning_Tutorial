# ContrastiveLearning_Tutorial
This repository contains all the material that will be presented in CL tutorial at IEEE BigData'23 conference and CODS-COMAD'24.
The motivation of this tutorial is to extend CL's focus towards other modalitites such as time series, graph and tabular in addition to vision, text and audio.
This repository contains example jupyter notebooks where the reader can implement contrastively trained models for time series and tabular data. 
The datasets are also included in the repository in 'Small_datasets' directory.

ts2vec folder is attributed to the authors of [TS2VEC paper](https://github.com/yuezhihan/ts2vec).
scarf folder is adapted from the pytorch implementation of [SCARF](https://github.com/clabrugere/pytorch-scarf/tree/master).
Database_matching folder is adapted from an application of CL to [integrate EHR datasets](https://github.com/sandhyat/KMFChimericE_SchMatch).

In addition to providing implementation of CL models, we also demonstrate:
1) optimizing contrastive loss as a discriminatory task by tracking the rank across epochs. This provides us an idea as to how easy or hard the learning task is.
2) modality gap which can be observed in all the multimodal contrastive learning training [Mind the gap](https://arxiv.org/pdf/2203.02053.pdf). In particular, we demonstrate this in the context of database matching where two datasets divided into known mapped and unmapped columns are used for CL and the goal is to match the unmapped columns from the two datasets.


## Code files
1) 'CL_for_TimeseriesDataset.ipynb': time series contrastive representation learning example using TS2VEC.
2) 'CL_for_TabularDataset.ipynb': tabular contrastive representation learning example using SCARF.
3) 'ModalityGap_in_MultiviewCL.ipynb': database matching example with CL that shows the existence of a modality gap..
4) 'CL_Disc_rank_metric.ipynb': example showing how the ranked temperature normalized similarity of positive pairs can be used as an indicator of learning path. 
5) 'CLAP_model.ipynb': example showing the working of contrastive audio-text model with Huggingface implementation.

## Datasets
The datasets in the folder Small_datasets are open source. One of them is a synthetic dataset drawn from 20-D multivariate Gaussian distribution.