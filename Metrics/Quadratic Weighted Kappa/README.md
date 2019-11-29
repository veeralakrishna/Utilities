## Quadratic Weighted Kappa
-------

Quadratic Weighted Kappa, which measures the agreement between two outcomes. This metric typically varies from 0 (random agreement) to 1 (complete agreement). In the event that there is less agreement than expected by chance, the metric may go below 0.

The outcomes in this competition are grouped into 4 groups (labeled accuracy_group in the data):

 - 3: the assessment was solved on the first attempt
 - 2: the assessment was solved on the second attempt
 - 1: the assessment was solved after 3 or more attempts
 - 0: the assessment was never solved


The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix O is constructed, such that Oi,j corresponds to the number of installation_ids i (actual) that received a predicted value j. An N-by-N matrix of weights, w, is calculated based on the difference between actual and predicted values:

https://www.codecogs.com/latex/eqneditor.php

w_{i,j} = \frac{\left(i-j\right)^2}{\left(N-1\right)^2}

![](https://latex.codecogs.com/gif.latex?w_%7Bi%2Cj%7D%20%3D%20%5Cfrac%7B%5Cleft%28i-j%5Cright%29%5E2%7D%7B%5Cleft%28N-1%5Cright%29%5E2%7D)


An N-by-N histogram matrix of expected outcomes, E, is calculated assuming that there is no correlation between values.  This is calculated as the outer product between the actual histogram vector of outcomes and the predicted histogram vector, normalized such that E and O have the same sum.

From these three matrices, the quadratic weighted kappa is calculated as: 



\kappa=1-\frac{\sum_{i,j}w_{i,j}O_{i,j}}{\sum_{i,j}w_{i,j}E_{i,j}}.


![](https://latex.codecogs.com/gif.latex?%5Ckappa%3D1-%5Cfrac%7B%5Csum_%7Bi%2Cj%7Dw_%7Bi%2Cj%7DO_%7Bi%2Cj%7D%7D%7B%5Csum_%7Bi%2Cj%7Dw_%7Bi%2Cj%7DE_%7Bi%2Cj%7D%7D.)
