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

db = gym.make('HovorkaCambridge-v0')
db_test = gym.make('HovorkaCambridge-v0')


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

max_episodes = 800
max_steps = 72
batch_size = 32

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
      


#%%

model = DQN(state_space, action_space)
model_target = DQN(state_space, action_space)

loss = nn.MSELoss()
optimizer = optim.Adam(params = model.parameters(), lr = learning_rate)

buffer = ReplayBuffer(10000)

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
def learn(batch_size, gamma):
      if buffer.size() < batch_size:
            return
      
      cs_batch, a_batch, r_batch, d_batch, ns_batch = buffer.sample(batch_size)
      
      Q_target = th.zeros((batch_size, action_space))
      
      for b in range(batch_size):
            Q_new = model_target(th.from_numpy(ns_batch[b]).float())
            maxQ_new = th.max(Q_new.data)
      
            Q_target[b] = model_target(th.from_numpy(cs_batch[b]).float())
            Q_target[b, a_batch[b]] = r_batch[b]
      
            if d_batch[b] == False:
                  Q_target[b, a_batch[b]] += gamma * maxQ_new
      
      Q_batch = model(th.from_numpy(cs_batch).float())
      
      train_loss = loss(Q_batch, Q_target)
      
      # Minimize the loss and backpropagate
      optimizer.zero_grad()
      train_loss.backward()
      optimizer.step()
      
      return train_loss
      
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
            #state = db.reset()
            # Reward sum, loss sum and time steps reset
            reward_all = 0
            t = 0
            
            # Time step loop
            while t < steps:
                  # Q values from local network, given current state
                  Q = model(th.from_numpy(state).float())
                  # Action selection
                  action = choose_action(Q,epsilon)
                  # Agent step
                  next_state, reward, done, _ = db.step(action)
                  
                  # Add transitions to the buffer memory
                  buffer.add(state, action, reward, done, next_state)
                  
                  # Training loss
                  train_loss = learn(batch_size, gamma)
                  
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
            
            # Update target network
            if epi % 10 == 0:
                  target_update(model, model_target, sigma)
            
            # Save time steps, reward, loss and epsilon into lists
            step_list.append(t)
            reward_list.append(reward_all)
            #loss_list.append(l/t)
            epsilon_list.append(epsilon)
            
            if epi % 100 == 0:
                  print('Episode: {}'.format(epi))
                  

#%%

train(max_episodes, max_steps, eps_init)


#%%
#-------------------------
## SAVE TRAINED MODEL PARAMETERS ##

filepath = './Documents/Python/Network parameters/checkpoint_hc.pth'

th.save(model.state_dict(), filepath)

# =============================================================================
# checkpoint = {'model': model,
#               'state_dict': model.state_dict(),
#               'optimizer': optimizer.state_dict()}
# th.save(checkpoint, filepath)
# =============================================================================


#%%
#-------------------------
## LOAD TRAINED MODEL PARAMETERS ##

# =============================================================================
# def load_checkpoint(filepath):
#       checkpoint = th.load(filepath)
#       model = checkpoint['model']
#       model.load_state_dict(checkpoint['state_dict'])
#       optimizer.load_state_dict(checkpoint['optimizer'])
#       
#       for p in model.parameters():
#             p.requires_grad = False
#       
#       model.eval()
#       return model
# model = load_checkpoint(filepath)
# =============================================================================
      
model.load_state_dict(th.load(filepath))

#%%

db_test.reset();
state = db_test.reset()

for i in range(72):
      q = model(th.from_numpy(state).float())
      a = choose_action(q,0)
      action_list.append(a)
      
      s,r,d,i = db_test.step(a)
      
    

db_test.render()
plt.title('Blood Glucose over Time')
plt.xlabel('Time')
plt.ylabel('Blood glucose')



#%%
#------------------------- 
## ACTION PLOT ##

plt.plot(action_list)
plt.yticks((0,1,2))
plt.title('Actions taken over time')
plt.xlabel('Time steps')
plt.ylabel('Actions')



#%%
window = int(max_episodes/10)

plt.plot(pd.Series(step_list).rolling(window).mean())
plt.title('Step Moving Average ({}-episode window)'.format(window))
plt.xlabel('Episode')
plt.ylabel('Moves')

#%%

plt.plot(pd.Series(reward_list).rolling(window).mean())
plt.title('Reward Moving Average ({}-episode window)'.format(window))
plt.xlabel('Episode')
plt.ylabel('Reward')


#%%

plt.plot(epsilon_list)
plt.title('Random Action Parameter')
plt.xlabel('Episode')
plt.ylabel('Chance of Random Action')























