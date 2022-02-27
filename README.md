
# Adaptive rule based skin detector
Detecting human skin using thresholds.  

**Original paper:** N. Brancati, G. De Pietro,M. Frucci, and L. Gallo. “Human skin detection through correlation rules between the YCb and YCr subspaces based on dynamic color clustering”. Computer Vision and Image Understanding 155, 2017, pp. 33–42.


# Skin detection algorithm

A skin detection method based on a dynamic generation of the skin cluster range in the YCbCr color space by taking into account the lighting conditions. The method is based both on the identification of skin color clusters in the YCb and YCr subspaces and on the definition of correlation rules between the skin color clusters; it is efficient in terms of computational effort.


# How to build and use
1. Load dependencies (in a virtual env)
1. Start docker with up command
1. Move the datasets in a folder named `dataset`
1. Run `dyc_batch_py.py <mode>`


# Credits

Credits to the original version authors: 
https://github.com/nadiabrancati/skin_detection/

