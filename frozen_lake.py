# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:24:36 2020

@author: Emin
"""

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v0")
env.render()

"""
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    
"""

# Q table   

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyper parameter

alpha = 0.8
gama = 0.95
epsilon = 0.1

# Plotting Metrix

reward_list = []
hole_list = []

episode_number = 50000

for i in range(1,episode_number):
    
    # initialize environment
    state = env.reset()
    
    reward_count = 0
    hole = 0
    
    while True:
        
        # exploit vs explore to find action
        # %10 explore , %90 exploit 
        if random.uniform(0,1) < 0.1:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            
        # action process and take rewqrd / abservation
        
        next_state, reward, done, _ = env.step(action)
        
         # Q learning fuction
        
        old_value = q_table[state, action] # old value
        
        next_max = np.max(q_table[next_state]) # next_max
        next_value = (1 - alpha) * old_value + alpha * (reward + gama * next_max)
        
        # Q table update
        q_table[state, action] = next_value
        
        # update state 
        state = next_state
        
        # find wrong dropouts
        
        if reward == 0:
            hole += 1
            
        reward_count += reward
        
        if done:
            break
        
        if i%10 == 0:
            hole_list.append(hole)
            reward_list.append(reward_count)
            print("Episode {}, reward {}, wrong dropout {} ".format(i, reward_count, hole))
            
# %% visualize

fig , axs = plt.subplots(1,2)

axs[0].plot(reward_list)
axs[0].set_xlabel("episode")    
axs[0].set_ylabel("reward")

axs[1].plot(hole_list)
axs[1].set_xlabel("episode")    
axs[1].set_ylabel("hole")

axs[0].grid(True)
axs[1].grid(True)

plot.show()  

        
        



























