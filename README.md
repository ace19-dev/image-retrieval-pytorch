# Image-Retrieval-PyTorch
- The leftmost is the query image. The remaining images are retrieved from gallery and ordered from left to right by similarity. 
The upper three row are correct retrieval and last row is incorrect retrieval.
![](assets/127_46.jpg)
![](assets/4371_8.jpg)
![](assets/2362_27.jpg)
![](assets/177_4.jpg)

- It might also be retrieved in the same category elaborately.
![](assets/1428_38.jpg)
![](assets/1168_38.jpg)
![](assets/1596_38.jpg)


## dataset
- https://www.kaggle.com/c/imaterialist-challenge-furniture-2018
- download_images.py (about 20GB) from https://www.kaggle.com/aloisiodn/python-3-download-multi-proc-prog-bar-resume Or use datasets/download_images.py

## References from 
- https://arxiv.org/abs/1812.00442
  - Deep Cosine Metric Learning for Person Re-Identification
- https://arxiv.org/abs/1807.00537
  - SphereReID: Deep Hypersphere Manifold Embedding for Person Re-Identification
  - learn a hypersphere manifold embedding
  - propose a convolutional neural network called SphereReID adopting Sphere Softmax
  - https://github.com/CoinCheung/SphereReID
- https://arxiv.org/abs/1801.09414
  - CosFace: Large Margin Cosine Loss for Deep Face Recognition
- https://github.com/layumi/Person_reID_baseline_pytorch
- https://github.com/KaiyangZhou/deep-person-reid
