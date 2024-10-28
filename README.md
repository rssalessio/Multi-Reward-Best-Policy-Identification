
# Multiple Rewards Best Policy Identification

  

This repository hosts the code accompanying the NeurIPS24 paper "Multiple Rewards Best Policy Identification". Our study investigates the exploration problem in Reinforcement Learning (RL) in presence of multiple rewards.

  

**Authors**: Alessio Russo, Filippo Vannella

**License**: MIT

**Additional license info**:
- The `Riverswim, ForkedRiverswim, DeepSea` environments; and other files (`mdp.py`, `cutils.pyx`, `utils.py`) in the tabular folder were taken from the [Model Free Active Exploration repository](https://github.com/rssalessio/ModelFreeActiveExplorationRL), licensed under the MIT license.\

- The code of `DBMR-BPI` was originally adapted from `DBMF-BPI` in [Model Free Active Exploration repository](https://github.com/rssalessio/ModelFreeActiveExplorationRL), licensed under the MIT license.

- the `CartPoleSwingUp` environment, in `continuous/envs`, was taken from [Model Free Active Exploration repository](https://github.com/rssalessio/ModelFreeActiveExplorationRL) which originally was taken from the BSuite repository (DeepMind Technologies Limited) with APACHE 2.0 license.\

- The code for `APT, RND, Disagreement` was taken from the [URLB repository](https://github.com/rll-research/url_benchmark/tree/main/agent), licensed under the MIT license.\

Changes and additions to those files are licensed under MIT.

  

## Requirements

  

To run the examples you need atleast Python 3.11 and the following libraries installed: `numpy scipy cvxpy mosek torch matplotlib notebook tqdm seaborn pandas cython tabulate clarabel`. Additional libraries may be needed.

  

## How to run the simulations

  

### Example in Section B.4.1 (Non-convexity of the sub-optimality gap)

  

The numerical results in section B.4.1 of the manuscript, regarding the minimum sub-optimality gap, can be found in the Jupyter notebook `plot_convexity_gap.ipynb` in the root folder.

  

### Results for the tabular case

  

To plot the results for the tabular case you first need to run the simulations. To run the simulations, simply execute the Jupyter notebook `run_experiments_tabular.py` (make sure to adjust the parameters, like `NUM_PROCESSES`, etc...) in the root folder. Use the `plot_results_tabular.ipynb` notebook to plot the results.

  

Plots can be found in the folder `tabular\figures`.

  

### Results for continuous MDPs

  

#### Simulations

To run the simulations for the continous case, simply execute the file `continuous\collect_data.py` (make sure to adjust the parameters `NUM_PROCESSES` etc...). This file will collect all the data necessary to make the plots.

  

#### Plot results

  

The data analysis for Cartpole swing-up can be found in the Jupyter notebook `continuous\analzse_cartpole.ipynb`, while for DeepSea can be found in the notebook `continuous/analyze_deepsea.ipynb`.

  

### FAQ/Problems

  

- If you encounter problems plotting the results using a Jupyter notebook on Linux, remember to install the necessary latex packages `sudo apt-get install dvipng texlive-latex-extra texlive-fonts-recommended cm-super` (check here [https://stackoverflow.com/questions/11354149/python-unable-to-render-tex-in-matplotlib])

- Sometimes you need to create a `data` folder to make sure that the results are saved correctly. In general this folder is created automatically.
