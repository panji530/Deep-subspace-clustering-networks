# Deep-subspace-clustering-networks

Tensorflow implementation for our NIPS'17 paper:

Pan Ji*, Tong Zhang*, Hongdong Li, Mathieu Salzmann, Ian Reid. Deep Subspace Clustering Networks. in NIPS'17.

Due to several requests, an unpolished version of our codes is released here (Caution!! I'm not even sure that I uploaded the latest version...). A better commented/polished version may come later depending on my timelines.

The key thing to take care of during pre-training is early-stopping, which means you need stop training once the reconstructed images are reasonably good (visually). This would make the later fine-tuning step easier.

We also noted that there is an issue of numerical instablility for the current version due to the non-uniqueness of SVD. Any suggestions for resolving this are welcome. 

## Diagonal constraint on C (i.e., diag(C)=0)

If you use L-2 regularization on C, the diagonal constraint (diag(C)=0) is not necessary (cf. [this paper](https://www.researchgate.net/publication/261989058_Efficient_Dense_Subspace_Clustering)). If you use L-1 regularization, the diagonal constraint is then necessary to avoid trivial solutions.

The code released here is for L-2 regularization (i.e., DSC-Net-L2), so there is no diagonal constraint on C. However, implementing the diagonal constraint is easy. Assume the latent representation after the encoder is Z. Before passing to the decoder, you do:
```
tf.matmul((C-tf.diag(tf.diag_part(C))),Z)
```

## Dependencies

Tensorflow, numpy, sklearn, munkres, scipy.

## License

For academic usage, the code is released under the permissive BSD license. For any commercial purpose, please contact the authors.

## Contact
Pan Ji, peterji1990@gmail.com
