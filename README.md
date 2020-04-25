Convex Adversarial Layerwise Training (COLT)  <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>
=============================================================================================================

![overview](https://raw.githubusercontent.com/eth-sri/colt/master/media/colt.png)

COLT is a certified defense for neural networks against adversarial examples, presented in [our ICLR'20 paper](https://files.sri.inf.ethz.ch/website/papers/iclr2020-colt.pdf).
Our defense is based on a novel combination of adversarial training with existing provable defenses.
The key idea is to model neural network training as a procedure which includes both, the verifier and the
adversary.
In every iteration, the verifier aims to certify the network using convex relaxation while the adversary tries to
find inputs inside that convex relaxation which cause verification to fail.
This method produces state-of-the-art neural network in terms of certified robustness on the challenging CIFAR-10 dataset. 

# Setup

All of the required python packages are listed in `requirements.txt`.
We suggest to install the packages in a new virtual environment in the following way:

```
$ virtualenv venv
$ source venv/bin/activate
$ (venv) pip install -r requirements.txt
```

Additionally, we require Gurobi 9.0 to perform certification (Academic license is available for free).
Gurobi can be installed using:

```
$ wget https://packages.gurobi.com/9.0/gurobi9.0.0_linux64.tar.gz
$ tar -xvf gurobi9.0.0_linux64.tar.gz
$ cd gurobi900/linux64/
$ python setup.py install
```

# Training 

Next, we present scripts to train the networks from our paper and reproduce the results.
We use CIFAR-10 dataset and perform training as explained in our paper.
To train the networks please run:

```
$ scripts/train_cifar10_2_255
$ scripts/train_cifar10_8_255
```
for 2/255 and 8/255 perturbation, respectively.

# Certification

Similarly, to certify the networks and reproduce our results, please run:

```
$ ./scripts/certify_cifar10_2_255
$ ./scripts/certify_cifar10_8_255
```

Note that certifying all 10 000 examples from the test set can take several days.
An easy way to speed up certification would be to partition the certification of the test set across different CPUs and GPUs
by manipulating `start_idx` and `end_idx` arguments in the scripts.

Citing this work
---------------------
```
@inproceedings{balunovic2020Adversarial,
	title={Adversarial Training and Provable Defenses: Bridging the Gap},
	author={Mislav Balunovic and Martin Vechev},
	booktitle={International Conference on Learning Representations},
	year={2020},
	url={https://openreview.net/forum?id=SJxSDxrKDr}
}
```
Contributors
------------

* [Mislav Balunović](https://www.sri.inf.ethz.ch/people/mislav) - mislav.balunovic@inf.ethz.ch
* [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin) - martin.vechev@inf.ethz.ch

License and Copyright
---------------------

* Copyright (c) 2020 [Secure, Reliable, and Intelligent Systems Lab (SRI), ETH Zurich](https://www.sri.inf.ethz.ch/)
* Licensed under the [Apache License](http://www.apache.org/licenses/)



