# Deep-subspace-clustering-networks

Tensorflow implementation for our NIPS'17 paper:

Pan Ji*, Tong Zhang*, Hongdong Li, Mathieu Salzmann, Ian Reid. Deep Subspace Clustering Networks. in NIPS'17.

Due to several requests, an unpolished version of our codes is released here (Caution!! I'm not even sure that I uploaded the latest version...). A better commented/polished version may come later depending on my timelines.

The key thing to take care of during pre-training is early-stopping, which means you need stop training once the reconstructed images are reasonably good (visually). This would make the later fine-tuning step easier.

We also noted that there is an issue of numerical instablility for the current version due to the non-uniqueness of SVD. Any suggestions for resolving this are welcome. 

Dependencies:

Tensorflow, numpy, sklearn, munkres, scipy.

License

For academic usage, the code is released under the permissive BSD license. For any commercial purpose, please contact the authors.

Contact: Pan Ji, peterji1990@gmail.com
