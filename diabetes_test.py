#%%
#-------------------------
## IMPORTING MODULES ##

import numpy as np
import gym
import seaborn as sns
sns.set()

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import torchvision.transforms as T
from torch.autograd import Variable

from pylab import plot, figure, title, show, ion, legend, ylim, subplot

#%%
#------------------------- 
## CREATE ENVIRONMENT ##

from gym.envs.diabetes.hovorka_model import hovorka_parameters

env = gym.make('HovorkaCambridge-v0')

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
max_steps = 100

# Initialize memory
step_list = []
reward_list = []
loss_list = []
epsilon_list = []

state_space = db.observation_space.shape[0]
action_space = db.action_space.n

learning_rate = 0.01

db.reset()



#%%

for i in range(72):

    # Step for the minimal/hovorka model
    s, r, d, i = db.step(int(1))
    # Saved for possible printing
    # blood glucose
    bg.append(env.env.simulation_state[4])
    # cont glucose monitor
    cgm.append(env.env.simulation_state[-1] * env.env.P[12])
    # reward
    reward.append(r)
    

db.render()


#%%
#-------------------------
## ONE HOT ENCODING ##

def OHE(x, l):
    x = th.LongTensor([[x]])
    one_hot = th.FloatTensor(1,l)
    return one_hot.zero_().scatter(1,x,1)



#%%
    
class OracleQ(nn.Module):
      def __init__(self):
            super(OracleQ, self).__init__()
            
            self.fc1 = nn.Sequential(
                        nn.Conv1d(256, 256,1),
                        nn.BatchNorm1d(256),
                        nn.ReLU()
                        )
            
            self.fc2 = nn.Sequential(
                        nn.Conv1d(256, 256, 1),
                        nn.BatchNorm1d(256),
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

model = nn.Sequential(nn.Linear(state_space, action_space))
model_target = nn.Sequential(nn.Linear(state_space, action_space))

loss = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)

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
      Q_target = model_target(th.from_numpy(state).float())
      Q_target[action] = th.FloatTensor(reward) + th.mul(maxQ_new, gamma)
      
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
                  
#%%

train(max_episodes, max_steps, eps_init)               
                  
      


            
            



 
            

























