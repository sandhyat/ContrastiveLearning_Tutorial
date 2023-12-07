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


**Multimodal Contrastive learning** 

1) [ConVIRT](https://arxiv.org/pdf/2010.00747.pdf) 

2) [CLIP](https://arxiv.org/pdf/2103.00020.pdf) 

3) [UnCLIP](https://cdn.openai.com/papers/dall-e-2.pdf) 

4) [CMCR](https://arxiv.org/pdf/2305.14381.pdf) 
