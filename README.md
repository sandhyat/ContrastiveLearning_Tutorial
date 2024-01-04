# ContrastiveLearning_Tutorial
This repository contains all the material that will be presented in CL tutorial at IEEE BigData'23 conference and CODS-COMAD'24.
The motivation of this tutorial is to extend CL's focus towards other modalitites such as time series, graph and tabular in addition to vision, text and audio.
This repository contains example jupyter notebooks where the reader can implement contrastively trained models for time series and tabular data. 
The datasets are also included in the repository in 'Small_datasets' directory.

The slides for CODS-COMAD tutorial are [here](https://github.com/sandhyat/ContrastiveLearning_Tutorial/blob/main/CODS-COMAD2024_CL_tutorial.pdf.)

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

## References

**Loss functions**

1) [NCE 2010](http://proceedings.mlr.press/v9/gutmann10a.html)
2) [Triplet Loss](https://arxiv.org/pdf/1503.03832.pdf) 
3) [N-pair](https://papers.nips.cc/paper/2016/file/6b180037abbebea991d8b1232f8a8ca9-Paper.pdf) 
4) [Lifted Structured Loss](https://arxiv.org/pdf/1511.06452.pdf) 
5) [Soft NN Loss](http://proceedings.mlr.press/v97/frosst19a/frosst19a.pdf) 
6) [InfoNCE](https://arxiv.org/pdf/1807.03748v2.pdf) 
7) [NTXent](https://arxiv.org/abs/2002.05709)

**Theory**

1) [InfoMIN: label preserving](https://proceedings.neurips.cc/paper/2020/file/4c2e5eaae9152079b9e95845750bb9ab-Paper.pdf) 
2) [Understanding CL via Alignment and Uniformity](https://arxiv.org/pdf/2005.10242.pdf) 
3) [Entropy and Reconstruction for lower bounding MI between learnt Representation](https://arxiv.org/pdf/2307.10907.pdf) 
4) [Feature dropout: label destroying](https://arxiv.org/pdf/2212.08378.pdf) 
5) [CL and inductive biases](https://arxiv.org/abs/2202.14037)

**Batch construction strategies**

1) [CL as instance discrimination: Memory bank](https://arxiv.org/pdf/1805.01978.pdf) 
2) [Momentum Contrast MoCo: Queue ](https://arxiv.org/pdf/1911.05722.pdf) 
3) [NNCLR: Nearest Neighbour support set](https://arxiv.org/pdf/2104.14548.pdf) 
4) [CL with hard negative samples](https://openreview.net/pdf?id=CR1XOQ0UTh-)

**Augmentation**

1) [Text augmentations CERT ](https://arxiv.org/pdf/2005.12766.pdf) 
2) [TSTCC- Time series augmentation](https://arxiv.org/pdf/2208.06616.pdf) 
3) [CLOCS- Times series augmentation](https://arxiv.org/pdf/2005.13249.pdf) 
4) [TFC- Time series augmentation](https://arxiv.org/abs/2206.08496)
5) [TS2VEC- Time series augmentation](https://arxiv.org/pdf/2106.10466.pdf) 
6) [Finding order in Chaos- Time series augmentation](https://siplab.org/papers/neurips2023-chaos.pdf) 
7) [SubTab- Tabular augmentation](https://browse.arxiv.org/pdf/2110.04361.pdf) 
8) [GraphCL- Graph augmentation](https://proceedings.nips.cc/paper/2020/file/3fe230348e9a12c13120749e3f9fa4cd-Paper.pdf) 
9) [Text augmentation in embedded space](https://arxiv.org/pdf/2012.07280.pdf) 
10) [Graph augmentation in embedded space](https://arxiv.org/pdf/2112.08679.pdf) 

**Modality-specific training strategies**

1) [CL + Curriculum learning in text](https://arxiv.org/pdf/2109.05941.pdf)
2) [CL in Student Teacher Framework in text](https://arxiv.org/pdf/2111.04198.pdf)
3) [Neighborhood CL in time series](https://arxiv.org/pdf/2106.05142.pdf)
4) [Mixing up CL for time series](https://arxiv.org/abs/2203.09270)
5) [Object level CL in structures world models for graphs](https://arxiv.org/pdf/1911.12247.pdf)
6) [SCARF for tabular data](https://browse.arxiv.org/pdf/2106.15147.pdf)
7) [SAINT: dual attention based for tabular data](https://browse.arxiv.org/pdf/2106.01342.pdf)
8) [TransTab for multiple sets of tabular data](https://browse.arxiv.org/pdf/2205.09328.pdf)

**Multimodal Contrastive learning** 

1) [ConVIRT](https://arxiv.org/pdf/2010.00747.pdf) 
2) [CLIP](https://arxiv.org/pdf/2103.00020.pdf)
3) [UnCLIP](https://cdn.openai.com/papers/dall-e-2.pdf)
4) [CLAP-improved for audio](https://arxiv.org/pdf/2211.06687.pdf)
5) [CMCR](https://arxiv.org/pdf/2305.14381.pdf)
6) [FactorCL: an MI based perspective for multimodal CL](https://arxiv.org/pdf/2306.05268.pdf)
7) [LiT: role of pretrained supervised models in multimodal CL](https://arxiv.org/pdf/2111.07991.pdf)
8) [3Towers: role of pretrained model with 3 encoders](https://arxiv.org/pdf/2305.16999.pdf)
9) [Mind the Gap: understanding the embedding gap and its role in CL](https://arxiv.org/pdf/2203.02053.pdf)
10) [Attacks on Multimodal CL training](https://openreview.net/pdf?id=iC4UHbQ01Mp)

**Loss tweaks to conventional contrastive loss**

1) [Debiased CL: Avoiding false negative arising from similar class](https://proceedings.neurips.cc/paper_files/paper/2020/file/63c3ddcc7b23daa1e42dc41f9a44a873-Paper.pdf)
2) [Decoupled CL: controlling the hardness of discriminatory task](https://arxiv.org/pdf/2110.06848.pdf)
3) [ExpCLR: using the additional features instead of augmentation](https://arxiv.org/pdf/2206.11517.pdf)

**Non-negative example based CL**

1) [BYOL:  addition of prediction module](https://arxiv.org/pdf/2006.07733.pdf)
2) [SimSiam: removing projection head s and adding stop gradient](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Exploring_Simple_Siamese_Representation_Learning_CVPR_2021_paper.pdf)
3) [BarlowTwins: using correlation between the augmented reps](https://arxiv.org/pdf/2103.03230.pdf)
4) [SwaV: clustering and predicting reps from two networks](https://arxiv.org/pdf/2006.09882.pdf)
5) [Divide and Cluster: Mix of contrast and cluster](https://arxiv.org/pdf/2105.08054.pdf)
