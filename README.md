# Deep Clustering Network modified with a U-Net

This is a modification of PyTorch reimplementation of Deep Clustering Network (DCN) (Yang et al, 2017) by Yi-Xuan Xu (https://github.com/xuyxu) from https://github.com/xuyxu/Deep-Clustering-Network. 

#### Architecture
![image](https://user-images.githubusercontent.com/92257044/162743803-e15b12da-b8c5-4928-ab5c-05b5b866e6ce.png)

#### Results

||NMI |ARI |ACC |Runtime(s) |
|-|-|-|-|-|
|Original paper (Yang et al, 2017) |0.81 |0.75 |0.83 |- |
|DCN with convolutions and skip connections |0.425 |0.21 |0.39 |2870 |
|DCN with dense layers |0.45 |0.26 |- |3150 |
|DCN with convolutions |0.635 |0.48 |0.63 |3021|

#### Hyperparameter tuning
![image](https://user-images.githubusercontent.com/92257044/162744867-c1248148-f8a1-45b6-86e4-44499f0f68c5.png)
As λ increases
- Clustering performance of DCN with dense layers deteriorates slightly
- Clustering performance of DCN with convolutional layers improves slightly

#### Key takeaways

- Convolutional neural networks learn representations faster and better than fully connected neural networks (MLPs)
- Introducing skip connections in the convolutional neural network worsens the clustering performance
- Underlying trade-off between reconstruction and clustering, so optimization objective should be modified to decouple the two

#### Reference
* Yang et al. ''Towards K-means-friendly Spaces: Simultaneous Deep Learning and Clustering'', ICML-2017 (https://arxiv.org/pdf/1610.04794.pdf)
