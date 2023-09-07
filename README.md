# Structural_complexity
Structural complexity module

This repository provides a module to compute the structural complexity of two-dimensional images. 
In addition, we include a minimal example of how to use it.

Few important comments:

[1] The raw data corresponds to a 2 dimensional numpy array of size $N_s \times N$  where each line has $N = L \times L$ entries and corresponds to the pixels of a square image. There are $N_s$ samples.

[2] The code returns the structural complexity $\mathcal{C}_0$, $\mathcal{C}_1$, the dissimilarities $D_k$ and the values of $k$. 

This repository corresponds to the open source code of the manuscript in preparation and was developed by Eduardo Ibarra-García-Padilla, Stephanie Striegel, Richard T Scalettar, and Ehsan Khatami. The project is supported by grant DE-SC-0022311, funded by the U.S. Department of Energy, Office of Science.
