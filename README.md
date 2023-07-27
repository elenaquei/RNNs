Hopf bifurcations in RNNs

The code here provided is compatibkle with MAtlab 2019 a, and requires a working implementation of IntLab and MatCont.

This code allows the validation of Hopf bifrucation sin neuralODEs, with particular attention to the examples presented in the companion paper "Computer Validation of Neural Network Dynamics:\\ A First Case Study" by c. Kuehn and E. Queirolo.

A first time user should look at the examples provided in the paper and encoded in figure_generation for a first overview of the working of the code.
The definition of the network structure and activation functions in given in multilayerRNN_Hopf_validation
The backpropagation of the derivatives is implemented in allders
The validation of Hopf bifurcations is coded in algebraic_hopf_simple, where simple refers to the use of derivatives only up to order 3. This code is extremely general.
Additional files are left as utility functions for the expert user.

Comments and bug reports are welcome and can be forwarded directly to: elena.queirolo@tum.de
Use of this code should be cited by citing the companion paper.
This code is shared under the Creative Commons licence.
