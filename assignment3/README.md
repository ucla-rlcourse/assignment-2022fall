# Assignment 3: Reinforcement Learning Pipeline in Practice


*2022-2023 fall quarter, CS269 Seminar 5: Reinforcement Learning. Department of Computer Science at University of California, Los Angeles. Course Instructor: Professor Bolei ZHOU. Assignment Author: Zhenghao PENG.*

-----



> NOTE: **We only grade your assignment based on the report PDF file and python files. You can do whatever you like in this notebook! Running all cells in this jupyter notebook does not generate all reqiured figures in result.md**


(`README.md` and the intro of `assignment3.ipynb` are identical)

Welcome to the assignment 3 of our RL course!


Different from the previous assignment, this assignment requires you to finish codes mainly in python files instead of in a jupyter notebook. The provided jupyter notebook only walks you through the training and post-processing procedure based on those python files. 


This assignment is highly flexible. You are free to do hyper-parameter searching or improve your implementation by introducing new components. You are encouraged to survey materials in the internet that explain the mechanisms of these classic algorithms. As a start point, [Spinning Up](https://spinningup.openai.com/en/latest/) might be a good choice.



### Grading Scheme

We required you to finish the following tasks:

* Task 1: Implement TD3Trainer **(30/100 points)**
* Task 2: Implement PPOTrainer **(20/100 points)**
* Task 3: Implement GAILTrainer **(20/100 points)**
* Task 4: Conduct RL generalization experiments on one of the algorithms implemented **(30/100 points)**
* Task 5: Conduct RL generalization experiments on another one of the algorithms implemented **(Bonus 20 points)**
* Task 6: Conduct generalization experiments on GAIL that boosts the test performance. You can train and plug in a better PPO agent as the expert or adjust the hyper-pameters. The best agent in these experiments should has > 200 reward. **(Bonus 20 points)**




Therefore the possible maximum grade will be 140 points.


### Summary of Experiments

**Group 1: Baselines**

1. TD3 in Pendulum-v0
2. TD3 in MetaDrive-Tut-Easy-v0
3. PPO in CartPole-v0
4. PPO in MetaDrive-Tut-Easy-v0
5. GAIL in MetaDrive-Tut-Easy-v0

**Group 2: RL Generalization Experiments**

Choose one of TD3, PPO or GAIL, conduct generalization experiments (see "Training and test environments"). 


Bonus 20 points: Run generalization experiments on another algorithm that you are not chosen in above experiment. You need to train 6 agents in this case.




### Deliverables

The exported **PDF file** of the `result.md`: You will use the provided `result.md` and attach the required figures in it and then generate a PDF file. Please visit the `result.md` for details.

Compress all files in `assignment3` and prepares a **ZIP file**. No rar or other format.

You need to submit **both the zip file and the pdf file** to bruinlearn.


### File structure

You need to pay attention to the files below:

1. `train.py` - Train scripts for PPO in CartPole-v0 and MetaDrive. Please implement `train`.
4. `core/ppo_trainer.py` - PPO algorithm. Please implement `compute_action` and `compute_loss`.
5. `core/buffer.py` - Supporting data structure for PPO (GAE is implemented here). Please implement `compute_returns` for PPO.
6. `core/td3_trainer.py` - File which can be directly run to train TD3. Please implement TD3 here.
7. `assignment3.ipynb` - Useful jupyter notebook to walk you through the whole assignment. Unlike previous work, you are not required to fill anything in this notebook (but you can use it to do whatever you like). 
8. `result.md` - A **template** file for your final submission. You need to **generate a PDF file** based on it after attaching images in it. 
9. `[train|eval]_[ppo|gail|td3].py` and `[ppo|gail|td3]_generalization_[eval|train].sh` - Reference files that can be used to debug quickly or conduct formal experiments in batch.



### Training and test environments

We prepared a set of pre-defined MetaDrive environments to train your agents. 

The first is `MetaDrive-Tut-Easy-v0`, which only has one map with only a straight road.

We also prepare a set of environments for RL generalization experiments.

Concretely, we will use a training environment to train the RL agent and 
use a separate test environment `MetaDrive-Tut-Test-v0` to run the RL agent.
You can choose on of those training environments: 

* `MetaDrive-Tut-1Env-v0`
* `MetaDrive-Tut-5Env-v0`
* `MetaDrive-Tut-10Env-v0`
* `MetaDrive-Tut-20Env-v0`
* `MetaDrive-Tut-50Env-v0`
* `MetaDrive-Tut-100Env-v0`

Those training environment contain [1, 5, 10, 20, 50, 100] unique traffic scenarios, respectively. And the test environment contains 20 unique traffic scenarios which are unique to those in the training environments. 

By training a set of RL agents in different training environments and test them in the same test environment, we can examine the influence of the diversity of scenarios in the training environments. We expect to see that when using a training environment with more diverse data, the test performance of the trained agent should be improved, meaning it has better generalization ability.


**Please note that each MetaDrive can only run in single process. If you encounter any environmental error, please firstly try to restart the notebook and rerun the cell.**



### Notes

1. We use multi-processing to enable asynchronous data sampling. Therefore in many places the tensors have shape `[num_steps, num_envs, num_feat]`. This means that `num_envs` environments are running concurrently and each of them samples a fragment of trajectory with length `num_steps`. There are totally `num_steps*num_envs` transitions are generated in each training iteration.


2. Each process can only have a single MetaDrive environment. 


3. The jupyter notebook is used for tutorial and visualization. It is optional for you to use the notebook to train agents or visualize results.


### Colab supporting

Though we use multiple files to implement algorithms, we can still leverage Colab to access computing resources. 

* Step 1: Create a folder in your Google Drive root named `cs269`
* Step 2: Upload the files in `assignment3` folder such as `train_ppo.py` and the folder `core` to `cs269` folder in your Google Drive.
* Step 3: Run the last cell in Demo 1.

