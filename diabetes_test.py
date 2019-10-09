#%%
#-------------------------
## IMPORTING MODULES ##

import numpy as np
import gym
import seaborn as sns
sns.set()

import random
from collections import deque

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

from pylab import plot, figure, title, show, ion, legend, ylim, subplot
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

## Setting up matplotlib ##

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
plt.ion()

#%%
#------------------------- 
## CREATE ENVIRONMENT ##

from gym.envs.diabetes.hovorka_model import hovorka_parameters

db = gym.make('HovorkaCambridge-Sigurd-v0')


#%%
#------------------------- 
## INITIALIZE HYPERPARAMETERS ##

init_basal_optimal = 6.43
eps_init = 1.0
eps_end = 0.01
eps_decay = 0.995
gamma = 0.99
sigma = 1e-3

reward = []
bg = []
cgm = []

max_episodes = 1000
max_steps = 288

# Initialize memory
step_list = []
reward_list = []
loss_list = []
epsilon_list = []
action_list = []

state_space = db.observation_space.shape[0]
action_space = db.action_space.n

learning_rate = 0.01

db.reset()



#%%

for i in range(288):

    # Step for the minimal/hovorka model
    s, r, d, i = db.step(int(1))
    # Saved for possible printing
    # blood glucose
    bg.append(db.env.simulation_state[4])
    # cont glucose monitor
    cgm.append(db.env.simulation_state[-1] * db.env.P[12])
    # reward
    reward.append(r)
    

db.render()


#%%
#-------------------------
## REPLAY BUFFER ##

class ReplayBuffer:
      def __init__(self, buffer_size):
            self.buffer_size = buffer_size
            self.count = 0
            self.buffer = deque()
            
            
      def add(self, cs, a, r, d, ns):
            """Adds an experience to the buffer"""
            # cs: current state
            # a: action
            # r: reward
            # d: done
            # ns: next state
            experience = (cs, a, r, d, ns)
            if self.count < self.buffer_size:
                  self.buffer.append(experience)
                  self.count += 1
            else:
                  self.buffer.popleft()
                  self.buffer.append(experience)
      
      
      def size(self):
            return self.count
      
      
      def sample(self, batch_size):
            """Samples batch_size samples from the buffer, if the
            buffer contains enough elements. Otherwise, returns all elements"""
            
            batch = []
            if self.count < batch_size:
                  batch = random.sample(self.buffer, self.count)
            else:
                  batch = random.sample(self.buffer, batch_size)
            
            # Maps each experience in the batch in batches of current sates,
            # actions, rewards, dones and next states
            cs_batch, a_batch, r_batch, d_batch, ns_batch = list(map(np.array,
                                                                     list(zip(*batch))))
            return cs_batch, a_batch, r_batch, d_batch, ns_batch
      
      
      def clear(self):
            self.buffer.clear()
            self.count = 0    


#%%
#-------------------------
## DEEP Q-NETWORKS ##
            
class DQN(nn.Module):
      def __init__(self, state_space, action_space):
            super(DQN, self).__init__()
            
            self.layer = nn.Sequential(nn.Linear(state_space, action_space))
            
      def forward(self, x):
            return self.layer(x)
            
    
class OracleQ(nn.Module):
      def __init__(self):
            super(OracleQ, self).__init__()
            
            self.fc1 = nn.Sequential(
                        nn.Linear(state_space,256),
                        nn.BatchNorm1d(256),
                        nn.ReLU()
                        )
            
            self.fc2 = nn.Sequential(
                        nn.Linear(256, action_space),
                        nn.BatchNorm1d(action_space),
                        nn.ReLU()
                        )
            
      def forward(self, x):
            fc1 = self.fc1(x)
            return self.fc2(fc1)
            


class CNN_Q(nn.Module):
      def __init__(self, state_space, action_space):
            super(CNN_Q, self).__init__()
            
            
            
            

class GRU_Q(nn.Module):
      def __init__(self, state_space, action_space):
            super(GRU_Q, self).__init__()
            
            self.gru = nn.GRU(state_space, 128, 2)
            self.fc = nn.Linear(128, action_space)
            self.relu = nn.ReLU()
            
      def forward(self, x):
            gru, h = self.gru(x)
            fc = self.fc(gru)
            return self.relu(fc)
            

#%%
#-------------------------
## TRAINING MODEL ##
            
class Trainer():
      def __init__(self, network):
            # Initializing local and target networks
            if network == "DQN":
                  self.model = DQN(state_space, action_space)
                  self.model_target = DQN(state_space, action_space)
            elif network == "OracleQ":
                  self.model = OracleQ()
                  self.model_target = OracleQ()
            elif network == "GRUQ":
                  self.model = GRU_Q(state_space, action_space)
                  self.model_target = GRU_Q(state_space, action_space)
            
            # Loss function and optimizer
            self.loss = nn.MSELoss()
            self.optimizer = optim.Adam(params = self.model.parameters(),
                                        lr = learning_rate)
            
      
      # Action selection function
      def choose_action(self, Q, epsilon):
            # Random action selection
            if np.random.uniform() < epsilon:
                  return db.action_space.sample()
            # Greedy action selection
            else:
                  return np.argmax(Q.detach().numpy())
            
      
      # Learning function. Returns loss
      def learn(self, Q, curr_state, action, reward, next_state):
            # Q values from target network, based on next state
            Q_new = self.model_target(th.from_numpy(next_state).float())
            # Max Q value of Q_new
            maxQ_new = th.max(Q_new.data)
            
            # Initialize Q_target and update
            Q_target = self.model_target(th.from_numpy(curr_state).float())
            Q_target[action] = reward + th.mul(maxQ_new, gamma)
            
            return loss(Q, Q_target)
      
      
      # Target network update function
      def target_update(self, QNet_local, QNet_target, sigma):
            # Store parameters from both networks in a list
            theta = zip(QNet_local.parameters(), QNet_target.parameters())
            
            # Iterate through all the parameters in the list
            # and update the target weights according to:
            # θ_target = τ*θ_local + (1 - τ)*θ_target
            for theta_local, theta_target in theta:
                  thetaL_transformed = sigma * theta_local
                  thetaT_transformed = (1.0 - sigma) * theta_target
                  theta_transformed = thetaL_transformed + thetaT_transformed
                  
                  theta_target.data.copy_(theta_transformed)
            
      
      # Agent training function
      def train_agent(self, episodes, steps, epsilon):
            # Episode loop
            for epi in range(episodes):
                  # Initialize state
                  state = db.reset()
                  
                  # Reward sum, loss sum and time steps reset
                  r = l = t = 0
                  
                  # Time step loop
                  while t < steps:
                        # Q values from local network, given current state
                        Q = self.model(th.from_numpy(state).float())
                        # Action selection
                        action = self.choose_action(Q, epsilon)
                        # Agent step
                        next_state, reward, done, _ = db.step(action)
                        
                        # Training loss
                        loss_train = self.learn(Q, state, action, 
                                                reward, next_state)
                        # Loss sum
                        l += loss_train.item()
                        
                        # Minimize the loss and backpropogate
                        self.optimizer.zero_grad()
                        loss_train.backward()
                        self.optimizer.step()
                        
                        # Update target network weights
                        self.target_update(self.model,
                                           self.model_target, sigma)
                        
                        # Reward sum
                        r += reward
                        # Updating current state
                        state = next_state
                        # Incrementing time steps
                        t += 1
                        
                        # Break loop if done equals true
                        if done:
                              break
                        
                  # Update epsilon
                  epsilon = max(eps_end, eps_decay * epsilon)
                  
                  # Save time steps, reward, loss and epsilon into lists
                  step_list.append(t)
                  reward_list.append(r)
                  loss_list.append(l/t)
                  epsilon_list.append(epsilon)
                  action_list.append(action)
            
                  if epi % 100 == 0:
                        print('Episode: {}'.format(epi))
                        



model = DQN(state_space, action_space)
model_target = DQN(state_space, action_space)

loss = nn.MSELoss()
optimizer = optim.Adam(params = model.parameters(), lr = learning_rate)

# Action selection function
def choose_action(Q, epsilon):
      # Random action selecction
      if np.random.uniform() < epsilon:
            return db.action_space.sample()
      # Greedy action selection
      else:
            
            action = np.argmax(Q.detach().numpy())
            return action
      
# Learning function. Returns loss 
def learn(Q, curr_state, action, reward, next_state):
      # Q values from target network, based on next state
      Q_new = model_target(th.from_numpy(next_state).float())
       # Max Q value of Q_new
      maxQ_new = th.max(Q_new.data)
      # Initialize Q_target and update
      Q_target = model_target(th.from_numpy(curr_state).float())
      Q_target[action] = reward + th.mul(maxQ_new, gamma)
      
      return loss(Q, Q_target)
      
# Target network update function
def target_update(QNet_local, QNet_target, sigma):
      # Store parameters from both networks in a list
      theta = zip(QNet_local.parameters(), QNet_target.parameters())
            
      # Iterate through all the parameters in the list
      # and update the target weights according to:
      # θ_target = τ*θ_local + (1 - τ)*θ_target
      for theta_local, theta_target in theta:
            thetaL_transformed = sigma * theta_local
            thetaT_transformed = (1.0 - sigma) * theta_target
            theta_transformed = thetaL_transformed + thetaT_transformed
            
            theta_target.data.copy_(theta_transformed)      
      
# Agent training function
def train(episodes, steps, epsilon):
      
      # Episode loop
      for epi in range(episodes):
            # Initiliaze state
            state = db.reset()
            # Reward sum, loss sum and time steps reset
            reward_all = 0
            l = 0
            t = 0
            
            # Time step loop
            while t < steps:
                  # Q values from local network, given current state
                  Q = model(th.from_numpy(state).float())
                  # Action selection
                  action = choose_action(Q,epsilon)
                  # Agent step
                  next_state, reward, done, _ = db.step(action)
                  
                  # Training loss
                  train_loss = learn(Q, state, action, reward, next_state)
                  
                  # Loss sum
                  l += train_loss.item()
                  
                  # Minimize the loss and backpropagate
                  optimizer.zero_grad()
                  train_loss.backward()
                  optimizer.step()
                  
                  # Update target network
                  target_update(model, model_target, sigma)
                  
                  # Reward sum
                  reward_all += reward
                  # Updating current state
                  state = next_state
                  # Incrementing time steps
                  t += 1
                  
                  # Break loop if done equals true
                  if done:
                        break
            
            # Update epsilon
            epsilon = max(eps_end, eps_decay * epsilon)
            
            # Save time steps, reward, loss and epsilon into lists
            step_list.append(t)
            reward_list.append(reward_all)
            loss_list.append(l/t)
            epsilon_list.append(epsilon)
            action_list.append(action)
            
            if epi % 100 == 0:
                  print('Episode: {}'.format(epi))
                  
#%%

#train(max_episodes, max_steps, eps_init)

Train = Trainer("DQN")
Train.train_agent(max_episodes, max_steps, eps_init)


#%%
#------------------------- 
## PLOTS ##

# Plotting window
window = int(max_episodes/10)      

plt.figure(figsize=[9,16])
      
plt.subplot(411)
plt.plot(pd.Series(step_list).rolling(window).mean())
plt.title('Step Moving Average ({}-episode window)'.format(window))
plt.xlabel('Episode')
plt.ylabel('Moves')

plt.subplot(412)
plt.plot(pd.Series(reward_list).rolling(window).mean())
plt.title('Reward Moving Average ({}-episode window)'.format(window))
plt.xlabel('Episode')
plt.ylabel('Reward')

plt.subplot(413)
plt.plot(pd.Series(loss_list).rolling(window).mean())
plt.title('Loss Moving Average ({}-episode window)'.format(window))
plt.xlabel('Episode')
plt.ylabel('Loss')

plt.subplot(414)
plt.plot(epsilon_list)
plt.title('Random Action Parameter')
plt.xlabel('Episode')
plt.ylabel('Chance of Random Action')

plt.tight_layout(pad=8)
plt.show()

#%%
#-------------------------
## TEST TRAINED MODEL ##

state = db.reset()

for i in range(288):
      q = model(th.from_numpy(state).float())
      a = choose_action(q, 0)
      s,r,d,i = db.step(a)
      
db.render()
plt.title('Blood Glucose over Time')
plt.xlabel('Time')
plt.ylabel('Blood glucose')

            
#%%
#------------------------- 
## ACTION PLOT ##

plt.plot(action_list[0:50])
plt.yticks((0,1,2))

#%%
#------------------------- 
## REWARD MOVING AVERAGE ##

plt.plot(pd.Series(reward_list).rolling(int(max_episodes/10)).mean())





























