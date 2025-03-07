# Deep learning for predicting molecule spectra.
Written as a special assignment in physics for Aalto Univesity CEST group.

Network originally designed and created in Theano by Kunal Ghosh.

Rewritten into PyTorch by Mathias Smeds, along with further hyperparameter optimisation.

Final submission at [link](./PHYS_special_assignment_mathiassmeds.pdf)

# Packages
1. conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
2. conda install scikit-learn

# Running instructions
```shell
stdbuf -oL -eL python3 spectroscopy_run_experiments.py ../../data/132k_16_opt_eV/spectra.npz ../../data/132k_16_opt_eV/coulomb.npz 1000 exp00 rmse 300 0.1 |& tee -a 0_1_output.txt
```
* `stdbuf -oL -eL` ensures that python uses linebuffering for output and error streams respectively.
* In addition to the the `stdbuf` one must flush the stdout and stderr in the training loop.
* By default 0.05 percent of total data (6627 datapoints) is used for test and validation sets.
* The last positional argument `0.1` in the example above indicates the percent of the total data (excluding test and validation set) that is used for training. 
* `|&` pipes both the stdout and stderr.
* `tee -a` appends everything that is piped to the text file (here `0_1_output.txt`) and also prints it out to stdout.
