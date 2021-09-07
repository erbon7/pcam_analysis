

In this study I compare different architectures of convolutional neural
networks and different hardware acceleration devices for the detection of
breast cancer metastasis tissue from microscopic images of sentinel lymph
nodes. Convolutional models with increasing depth are trained and tested on a
public data set of more than 300,000 images of lymph node tissue, on three
different hardware acceleration cards, using an off-the-shelf deep learning
framework. The impact of transfer learning, data augmentation and
hyperparameters fine-tuning are also tested. Hardware acceleration device
performance can improve training time by a factor of five to seven, depending
on the model used. On the other hand, increasing convolutional depth will
augment the training time by a factor of four to six times, depending on the
acceleration device used. Increasing the depth of the model, as could be
expected, clearly improves performance, while data augmentation and transfer
learning do not. Fine-tuning the hyperparameters of the model notably improves
the results, with the best model showing a performance comparable to
state-of-the-art models.

Pre-print available here: (https://arxiv.org/abs/2108.13661)
