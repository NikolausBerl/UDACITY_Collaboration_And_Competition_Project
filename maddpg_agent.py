import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic  

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e7)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

# UPDATE_EVERY_TIMESTEP = 10  # not used
NUM_UPDATES = 6

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize an Agent object.
        
        Params
        ======noise
            state_size (int): dimension of each state
            action_size (int): dimension of each actio5 
            random_setheed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        self.counter = 0

        
        self.gamma = 0.99
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, num_agents, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, num_agents, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, num_agents, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, num_agents, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        #self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def step(self, states, actions, act_2, rewards, next_states, dones, shared_ReplayBuffer, time_step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        
        #print("---- Step-Function: states: {}, actions: {}, rewards: {}, next_states: {}, dones: {}".format( states, actions, rewards, next_states, dones))
        shared_ReplayBuffer.add(states, actions, act_2, rewards, next_states, dones)
        
        # Learn, if enough samples are available in memory
        if len(shared_ReplayBuffer) > BATCH_SIZE:
            for _ in range(NUM_UPDATES): # Extra-Updates
                experiences = shared_ReplayBuffer.sample()
                self.learn(experiences, self.gamma)
     
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        #print("4.0- vor: self.actor_local.eval() ")
        self.actor_local.eval()
        #print("4.1- nach: self.actor_local.eval()")
        self.counter +=1
        with torch.no_grad():
            # Here we are proceeding the forward-pass of the actor
            #print("4.2- in Class Agent --> vor: self.actor_local(state).cpu().data.numpy(): param: state", state)
            action = self.actor_local(state).cpu().data.numpy()
            #print("4.3- in Class Agent -->nach: self.actor_local(state).cpu().data.numpy()", param: state, state)
     
            self.actor_local.train()
            if add_noise:
                print("add_noise == True")
                action += self.noise.sample()   
        return np.clip(action, -1, 1)
      
    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, act_2, rewards, next_states, dones = experiences
        #print("5- learn-Methode- type(actions): {}, actions.shape: {}, actions: {}".format(type(actions), actions.shape, actions))
        #print("5- learn-Methode- type(act_2): {}, act_2.shape: {}, act_2: {}".format(type(act_2), act_2.shape, act_2))
        
        
        # print("learn: len(states), len(actions), len(rewards), len(next_states), len(dones)",
        #       len(states), len(actions), len(rewards), len(next_states), len(dones))
        # mit len(states) etc wird die Batch_size angezeigt. --> Anzahl der Datensätzen pro Batsch.
        # Danach wird der komplette Datensatz an die NN übergeben und zusammen verarbeitet.
        # ---------------------------- update critic --------------------------------------------#
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        
        
        #Implict C-A-L-L of the forward of the critic ---------------------------------
        # actions_next = actions_next + act_2
        Q_targets_next = self.critic_target(next_states, actions_next, act_2)

        
        # Compute Q targets for current states (y_i
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        
        #Implict C-A-L-L of the forward of the critic ---------------------------------
        Q_expected = self.critic_local(states, actions, act_2)
        
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # gradient Clipping for the Critics, as recommended in the Benchmark-Implementation part.
        # torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)  

        self.critic_optimizer.step()
        
        # ---------------------------- update actor -------------------------------------------------#
        # Compute actor loss
        actions_pred = self.actor_local(states)
        
        #-------------------------------------------------------------------------------
        # Put the action of the other player to critic-input
        #actions_pred = np.stack((actions_pred, act_2), axis=0).flatten()
        
        actor_loss = -self.critic_local(states, actions_pred, act_2).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm(self.actor_local.parameters(), 1)  

        self.actor_optimizer.step()

        # ----------------------- update target networks ------------ #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)  
        

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    # def __init__(self, size, seed, mu=0., theta=0.18, sigma=0.1):
    def __init__(self, size, seed, mu=0., theta=0.14, sigma=0.25):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action","act_2", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, act_2, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, act_2, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        #random.sample(population, k)
        #Return a k length list of unique elements chosen from the population sequence or 
        #set. Used for random sampling without replacement.
        
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        act_2 = torch.from_numpy(np.vstack([e.act_2 for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, act_2, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)