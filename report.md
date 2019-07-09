
##### About the Tennis-Player Environment (2 Agents are playing as oponents)


##### Implemented Algorithm "Multi Agent Deep Deterministic Policy Gradient" - MADDPG

Multi-Agent rl-systems are able to solve targets together.
It is important to find an appropriate algorithm and reward-structure,
that the individual agents can solve the given problem together at best.
The behaviour of such agents can be between high cooperation and little competing and
little cooperation and high competing.

In real multi-agent systems, an agent dispose not simply over information of the
environment but also over information of the other agents.
Without information about the other agents, a RL-system is difficult to solve.
The reason for this is, that each agent develop his own policy.

As a result of this individual policy development the environment becomes "non-stationary".
This means, that one agent has difficulties to predict the others behaviour with simply environment information
One new and sucessful algorithm (from 2018) is the 
"Multi-Agent-Actor-Critik for Mixed Cooperative-Competitive Environments" 
algorithm, in short MADDPG.
As recommended, I implemend my solutiln by following the MAADDG-Algorithm according to the 
arxiv-science-paper [1706.02275](https://arxiv.org/pdf/1706.02275.pdf)

MADDP is an extension of the actor-critic-policy-gradient methode (DDPG).
For this I take  the DDPG-Algorithm of the udacity ddpg-Pendulum-project as a programming basis.

#### Functionaliy of the MADDPG-Algorithm
- During the execution-time

The learned policies can only use local environmental information.
There is no particular structure for a direct information-exchange between agents.

- How the agents get anyway information about the other agents ?

During training-time, the critic (of first agent) gets in addition to
the state-information (of first agent) also the actions of the other agents  as an input.
The critic (of first agent) himselves take this information to compute the q-value, which helps to
optimize the actors (of first agent) prediction.
With this structure, the actors policies are also shaped with information about the other
agents. 
This is a framework of a centralized training and decentralized execution.
Therefore the agents can be executed without the critics during execution time.
The non-stationary-problem can be avoided with this algorithm.
The following functional overview shows how a multi-agent-system with MADDP work.


##### Functional overview
![MADDPG Functional Overview](MADDPG_Science_Paper_Picture.png)

##### MADDPG pseudocode
![MADDPG Algorithm](MADDPG_PseudoCode.png)
(See Paper 02275 )

### Code-Basis
The used Code-Basis is the Pendulum DDPG-Example-Program.
First I made running the code by adapting it to the new 
double-jointed arm environment.

#### Neural Network Architecture

##### The Actor 

* Input-Layer with 33 (state size) 
* First hidden Layers  with the 400 notes (input 33 nodes of the state-space) and Relu-Activation Function
* Second hidden Layer with 300 notes input (input 400 notes) and Relu-Activation Function
* Output Layer sith 4 notes for continuous output values-Vector and tangh Activation Function

##### Critic Network
* Input-Layer with 33 (state size) 
* First hidden Layers  with the 400 notes (input 33 nodes of the state space) and Relu-Activation Function
* Second hidden Layer with 300 notes input (input 400 notes) and Relu-Activation Function
* Output Layer with 1  note for the action value


##### Code adaptation



#### Parameter-Search
The adaptation of the following Parameters leaded to stable net-learning:
* Increasing the BUFFER_SIZE from 1e5 to 1e6        --> replay buffer size
* Increasing the BATCH_SIZE from 128 to 512         --> minibatch size
* Increasing the WEIGHT_DECAY from 0 to 1e-6        --> L2 weight decay
* Decreasing the  LR_ACTOR from 1e-3 to 1e-4        --> learning rate of the actor 

The following parameter stayed unchanged
* GAMMA = 0.99            --> discount factor
* TAU = 1e-3              --> for soft update of target parameters
* LR_CRITIC = 1e-3        --> learning rate of the critic

Conclusion: 
The best influence for training the Network was the change of 
the BATCH_SIZE and the WEIGHT_DECAY


### Testing

#### Training result


#### Improvements
- Optimization of Hyperparameters
To just try to change parameters and then have a look on the results is not  effective 
and time consuming. Furthermore it isn't a target oriented proceeding.
It needs to work out an effective proceeding of the optimal parameter-finding.

#### Test and compare of different Algorithms




