# KDD 2020: Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions

This repo contains the code for the Reward interaction Inverse Propensity Scoring off-policy estimator proposed in 
the KDD 2020 paper [Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions](https://arxiv.org/pdf/2007.12986.pdf)
by James McInerney, Brian Brost, Praveen Chandar, Rishabh Mehrotra, Ben Carterette. 


This implementation uses Python Beam and Google's Dataflow for running experiments to make it easier to scale to large datasets. However, if you are interested in running a simple simulation experiment, 
you can do so using the following command (make sure to install the dependencies, see Environment Setup) 

```shell script
PYTHONPATH=./ python run.py [output_path]
```
The script generates two files for each run. Use the `analysis.ipynb` notebook to generate the plots
similar to the ones in the paper.




### Environment Setup

Create a new virtual environment with for your supported Python version. We recommend the use of [Anaconda](https://www.anaconda.com/distribution/)
for managing virtual environments. Create a new environment `conda create --name rips python=3.7` and switch to 
the environment using `conda activate rips`.  

```shell
$ pip install -r requirements.txt
```

### Reward interaction IPS
The implementation of the  Reward interaction IPS (RIPS) can be found in `rips/eval/offpolicy/rips.py` file.
