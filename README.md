
<img src="./attachments/udacityTennisPlayer.gif" width="80%" align="top-left" alt="" title="My Banana Agent" />

### Project 3: Collaboration and Competition
(realized with MADDPG-Algorithm)

[//]: # (Image References)

### Introduction
In this Project two RL-Agents are playing tennis together. They
control rackets to bounce a ball over a net
Each agent receives its own, local state-observation and creates his own action.

### Reward-Function:
If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

####  Target to achieve:
The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.


#### Installation & requirements

#### 1.) Installing the python Environment:

Please Follow the Instruction of the Udacity-DRLN-Repository: [click here](https://github.com/udacity/deep-reinforcement-learning#dependencies)


#### 2.) Installing the Unity-environment

Select your operating system and download the environment from one of the links below.

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

Then, place the file in the folder of your DRLND working repository, and unzip (or decompress) the file.

(For Windows users) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

#### 3.) Copy the developed project-files in your DRLND working directory:

3.1 Code-Files

- Tennis.ipynb
- Tennis_Player.ipynb
- model.py
- maddpg_agent.py

3.2 Network Weight-Files

If you want use the network weights of the given result, please copy also the following files for the actor and critic network in your working-directory: 

**Players with player-level 0.1:**
- score_01_player_1_checkpoint_actor.pth 
- score_01_player_1_checkpoint_critic.pth
- score_01_player_2_checkpoint_actor.pth 
- score_01_player_2_checkpoint_critic.pth

**Players with player-level 0.3**
- score_03_player_1_checkpoint_actor.pth 
- score_03_player_1_checkpoint_critic.pth
- score_03_player_2_checkpoint_actor.pth 
- score_03_player_2_checkpoint_critic.pth

**Players with player-level 0.5**
- score_01_player_5_checkpoint_actor.pth 
- score_01_player_5_checkpoint_critic.pth
- score_01_player_5_checkpoint_actor.pth 
- score_01_player_5_checkpoint_critic.pth




3.3 run one of the ipynb-Notebooks












    

