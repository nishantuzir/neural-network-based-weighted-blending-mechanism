# neural-network-based-weighted-blending-mechanism
This is a weighted blending machine implemented using a neural network. The advantage of using a neural network is that the weights assigned to the models for the final result is assigned by the neural network based on backpropagation.

OUTPUT FORMAT:
--------------
At the end of the training, the following values will be displayed:
confusion matrix
accuracy
precision
recall
f1 score

![alt text](https://github.com/nishantuzir/neural-network-based-weighted-blending-mechanism/blob/master/output.png)

PS: The module returns the neural network model, which inturn can be used to test on a given dataset.

USAGE:
------
$ python3 ./blending.py -t ./training.csv -s 0 -e 12   
OR    
$ python3 ./blending.py --train ./training.csv --start 0 --end 12

PS: --start and --end values should integer and not float!

if you need help, the following command would be useful:

$ python3 ./janf.py -h

DEPENDENCY:
-----------
python3.x

PACKAGES:
---------
The packages required for running the flowmeter, are provided in the 'requirements.txt' file.

The following python packages will be already be installed with the python3.x distibutions, if not, kindly install them:

1.sys

2.argparse

3.warnings
