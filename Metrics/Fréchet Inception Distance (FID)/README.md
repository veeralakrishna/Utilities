(https://www.kaggle.com/wendykan/demo-mifid-metric-for-dog-image-generation-comp/notebook) by @wendykan 
> 

# How to measure GAN performance?

In GANs, the objective function for the generator and the discriminator usually measures how well they are doing relative to the opponent. For example, we measure how well the generator is fooling the discriminator. It is not a good metric in measuring the image quality or its diversity. As part of the GAN series, we look into the Inception Score and Fréchet Inception Distance on how to compare results from different GAN models.

## Inception Score (IS)

IS uses two criteria in measuring the performance of GAN:

- The quality of the generated images, and
- **their diversity.**

Entropy can be viewed as randomness. If the value of a random variable x is highly predictable, it has low entropy. On the contrary, if it is highly unpredictable, the entropy is high. For example, in the figure below, we have two probability distributions p(x). p2 has a higher entropy than p1 because p2 has a more uniform distribution and therefore, less predictable about what x is.


![](https://cdn-images-1.medium.com/max/1600/1*RdIYRsqXxRAKwcjtxg6_kw.jpeg)

## Fréchet Inception Distance (FID)

In FID, we use the Inception network to extract features from an intermediate layer. Then we model the data distribution for these features using a multivariate Gaussian distribution with mean µ and covariance Σ. The FID between the real images x and generated images g is computed as:

![](https://cdn-images-1.medium.com/max/1600/1*tJmwViZesuFM89TcVN7J3A.png)

where Tr sums up all the diagonal elements.

&gt; Lower FID values mean better image quality and diversity.

FID is sensitive to mode collapse. As shown below, the distance increases with simulated missing modes.

![](https://cdn-images-1.medium.com/max/1600/1*8PzOnrzIeuM0E1unrFKLfg.png)

FID is more robust to noise than IS. If the model only generates one image per class, the distance will be high. So FID is a better measurement for image diversity. FID has some rather high bias but low variance. By computing the FID between a training dataset and a testing dataset, we should expect the FID to be zero since both are real images. However, running the test with different batches of training sample shows none zero FID.

![](https://cdn-images-1.medium.com/max/1600/1*D-XiZT9FdCWaA9jnyomsVw.png)

Also, both FID and IS are based on the **feature extraction** (the presence or the absence of features). Will a generator have the same score if the spatial relationship is not maintained?

## Precision, Recall and F1 Score

If the generated images look similar to the real images on average, the precision is high. High recall implies the generator can generate any sample found in the training dataset. A F1 score is the harmonic average of precision and recall.

In the Google Brain research paper “Are GANs created equal”, a toy experiment with a dataset of triangles is created to measure the precision and the recall of different GAN models.

![](https://cdn-images-1.medium.com/max/1600/1*0qc9oLuZxjeAqt4JBzPw2A.png)

## Kaggle Evaluation (MiFID)

Submissions are evaluated on MiFID (Memorization-informed Fréchet Inception Distance), which is a modification from Fréchet Inception Distance (FID).

 Kaggle calculates public and private MiFID scores with the same code, but with different pre-trained models and evaluation images. The public pre-train neural network is Inception, and the public images used for evaluation are the ImageNet Dogs (all 120 breeds). We will not be sharing what private model or dataset is used for the private MiFID score.
 
 A demo of our **MiFID** evaluation code can be seen here.
 
 Our workflow of computing the public/private MiFID is demonstrated below:
 
 **Check [Kaggle Evaluation Workflow](https://www.kaggle.com/c/generative-dog-images/overview/evaluation)**
 
 ![](https://storage.googleapis.com/kaggle-media/competitions/GAN/Kaggle%20GAN%20Diagram%20(3).png)

 I guess we need to find out more about the **mysterious NN** and **private image features**
 
 ## References
 - [GAN — How to measure GAN performance?](https://medium.com/@jonathan_hui/gan-how-to-measure-gan-performance-64b988c47732)
 
 - [Improved Techniques for Training GANs](https://arxiv.org/pdf/1606.03498.pdf)
 
 - [Are GANs Created Equal? A Large-Scale Study](https://arxiv.org/pdf/1711.10337.pdf)
 
 - [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/pdf/1706.08500.pdf)

