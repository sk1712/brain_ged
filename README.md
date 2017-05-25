# Finding correspondences between brain connectivity networks using graph edit distance

This code provides a Python implementation of constrained graph edit distance using the Hungarian algorithm (*Riesen and Bunke, 2009*) to identify correspondences between brain connectivity networks and obtain an estimate of their (dis)similarity.

<p align="center">
<img src="http://www.doc.ic.ac.uk/~sk1712/K_means_overview2.png" alt="overview" width="800"/>
</p>

Please cite the corresponding paper when using the code:

Ktena, S.I., Arslan, S., Parisot, S., Rueckert, D.: Exploring Heritability of Functional Brain Networks with Inexact Graph Matching. In: *IEEE International Symposium on Biomedical Imaging (ISBI)* (2017). [pdf](https://arxiv.org/pdf/1703.10062.pdf)

<p align="center">
<img src="http://www.doc.ic.ac.uk/~sk1712/correspondences_overview.png" alt="correspondences" width="500"/>
</p>

## Usage

The implementation of the GED algorithm, is provided by the *ged* package. A demo is provided in the iPython notebook *demo.ipynb*.

## Dependencies

This software uses Python 2.7 and depends on the following Python packages:

* scikit-learn
* numpy
* networkx
* nibabel
* nilearn
