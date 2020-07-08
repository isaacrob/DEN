# DEN

# I DON'T KNOW WHAT IS WRONG WITH THIS CODE BASE BUT PLEASE FOR NOW REFER TO MY ORIGINAL REPO WHICH I HAVE NOW OPEN SOURCED: https://github.com/isaacrob/SKN

We present  a visualization  algorithm based  on a  novel  unsupervised Siameseneural network training regime and loss function, called Differentiating EmbeddingNetworks (DEN). The Siamese neural network finds differentiating or similarfeatures between specific pairs of samples in a dataset, and uses these features toembed the dataset in a lower dimensional space where it can be visualized. Unlikeexisting visualization algorithms such as UMAP ort-SNE, DEN is parametric,meaning it can be interpreted by techniques such as SHAP. To interpret DEN, wecreate an end-to-end parametric clustering algorithm on top of the visualization,and then leverage SHAP scores to determine which features in the sample spaceare important for understanding the structures shown in the visualization basedon the clusters found. We compare DEN visualizations with existing techniqueson a variety of datasets, including image and scRNA-seq data.  We then showthat our clustering algorithm performs similarly to the state of the art despite nothaving prior knowledge of the number of clusters, and sets a new state of the arton FashionMNIST. Finally, we demonstrate finding differentiating features of adataset.

Link to paper: https://arxiv.org/abs/2006.06640

## Prereqs

We highly recommend running this with a GPU. It is VERY slow on CPU. Our package will automatically discover and use any CUDA-enabled GPU discovered.

To get started, make sure you have PyTorch, Sklearn, Numpy, progressbar2, and matplotlib (with its 3D toolkit) installed. Then, install SHAP (https://github.com/slundberg/shap) with pip:
`pip install shap`

## Usage

DEN.py shows example usage downloading and running the USPS dataset from torchvision. DEN can be run like a standard Sklearn classifier.

# MORE DOCUMENTATION TO COME SOON
