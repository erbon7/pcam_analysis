# pcam_analysis

This repository presents an exploratory analysis of the PCAM dataset.

PatchCamelyon (PCAM) is a public dataset of 327.680 images (96 x 96 px)
extracted from histopathologic microscope scans of lymph nodes sections.

Each image is annotated with a binary label indicating the presence of
metastatic (i.e. tumoral) tissue in the center of image (within a 32 x 32 px
frame). A positive value indicates that the center 32x32px region of the image
conatins at least one pixel of tumor tissue. Note that tumor tissue in the
outer region does not influence the value of the label. 

The original dataset is divided into a training set of 262.144 examples, a
validation set of 32768 examples and a test set of 32.768 examples.

For more details on the dataset, please see the original [PCAM
repository](https://github.com/basveeling/pcam) on github.

This dataset was released as part of a scientific work by PhD student Bas
Veeling:


**[1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation
Equivariant CNNs for Digital Pathology".
[arXiv:1806.03962](http://arxiv.org/abs/1806.03962)**

A citation of the original Camelyon16 dataset paper is appreciated as well:

**[2] Ehteshami Bejnordi et al. Diagnostic Assessment of Deep Learning
Algorithms for Detection of Lymph Node Metastases in Women With Breast Cancer.
JAMA: The Journal of the American Medical Association, 318(22), 2199–2210.
[doi:jama.2017.14585](https://doi.org/10.1001/jama.2017.14585)**





<!--
<img src="img/accuracy.png" width="200" align="center">

test
-->
