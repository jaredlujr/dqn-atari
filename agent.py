import os
import copy
from collections import namedtuple
from itertools import count
import math
import random
import numpy as np 
import time

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json

from wrappers import *
from replay_memory import ReplayMemory
from model import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Transition = namedtuple('Transion', ('state', 'action', 'next_state', 'reward','done'))

EPS_DECAY = 60000

class Agent(object):
    def __init__(self, 
                 env_name,
                 batch_size,
                 gamma,
                 lr,
                 target_update,
                 initial_memory,
                 memory_size,
                 dueling=False):
        # create environment
        env = gym.make(env_name)
        self.env = make_env(env)
        print('action space', self.env.action_space.n)
        
        # create networks
        if dueling:
            self.policy_net = DuelingDQN(n_actions=self.env.action_space.n).to(device)
            self.target_net = DuelingDQN(n_actions=self.env.action_space.n).to(device)
        else:    
            self.policy_net = ConvNN(n_actions=self.env.action_space.n).to(device)
            self.target_net = ConvNN(n_actions=self.env.action_space.n).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.criterion = nn.MSELoss()
        
        # setup optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1000,5000,10000,14000,20000,25000,30000], gamma=0.3)
        self.steps = 0
        
        # initialize replay memory
        self.memory = ReplayMemory(memory_size)
        
        # Hyperparam
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update = target_update
        self.lr = lr
        self.initial_memory = initial_memory

    def select_action(self, state):
        "epsilon greedy strategy"
        sample = random.random()
        # dynamic adjustment of epsilon
        eps_threshold = 0.01 + 0.4 * math.exp(-1. * self.steps / EPS_DECAY)
        self.steps += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return torch.argmax(self.policy_net(state.to(device))).cpu().detach()
        else:
            return torch.tensor([[random.randrange(self.env.action_space.n)]], device=device, dtype=torch.long)

    def optimize_model(self):
        """
        batch.state - tuple of all the states (each state is a tensor)
        batch.next_state - tuple of all the next states (each state is a tensor)
        batch.reward - tuple of all the rewards (each reward is a float)
        batch.action - tuple of all the actions (each action is an int)    
        batch.done - tuple of all the done state (each is a bool)
        """
        # Not enough batch
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        
        batch = Transition(*zip(*transitions))
        rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward))) 
        next_states = torch.cat([s for s in batch.next_state
                                        if s is not None]).to(device)
        state_batch = torch.cat(batch.state).to(device)
        reward_batch = torch.cat(rewards)
        
        state_action_values = self.policy_net(state_batch) 

        next_state_values = self.target_net(next_states).max(1)[0].detach()
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch  
        y = torch.FloatTensor(self.batch_size, self.env.action_space.n).zero_()
        y.copy_(state_action_values.detach())
        y = y.to(device)
        for idx in range(self.batch_size):
            y[idx][batch.action[idx]] = expected_state_action_values[idx]
        loss = self.criterion(state_action_values, y)

        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def get_state(self, obs):
        state = np.array(obs)
        state = state.transpose((2, 0, 1))
        state = torch.from_numpy(state)
        return state.unsqueeze(0)

    def train(self, n_episodes, render=False):
        curve_reward = []
        curve_loss = []
        for episode in range(n_episodes):
            obs = self.env.reset()
            state = self.get_state(obs)
            total_reward = 0.0
            total_loss = 0.0
            for t in count():
                action = self.select_action(state)
                if render:
                    self.env.render()
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                next_state = self.get_state(obs)
                reward = torch.tensor([reward], device=device)
                self.memory.push(state, action, next_state, reward, done)
                state = next_state
                if self.steps > self.initial_memory:
                    total_loss += self.optimize_model()
                    if self.steps % self.target_update == 0:
                       self.target_net.load_state_dict(self.policy_net.state_dict())
                if done:
                    break
            curve_reward.append(total_reward)
            curve_loss.append(total_loss)
            if episode % 5 == 0:
                print('Total steps:{} | Episode:{} | Steps:{} | Episode reward:{}'.format(self.steps, episode+1, t+1, total_reward))
        self.env.close()
        # plot the moving results
        plt.plot(curve_loss)
        plt.ylabel('Moving Loss each training episodes')
        plt.xlabel('Episodes')
        plt.savefig('dqn-loss.png')
        plt.close()

        plt.plot(curve_reward)
        plt.ylabel('Moving Reward each training episodes')
        plt.xlabel('Episodes')
        plt.savefig('dqn-rewards.png')
        plt.close()
        
    def test(self, n_episodes=8, render=True):
        for episode in range(n_episodes):
            obs = self.env.reset()
            state = self.get_state(obs)
            episode_reward = 0
            for t in count():
                action = self.select_action(state)
                if render:
                    self.env.render()
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward
                next_state = self.get_state(obs)
                state = next_state
                if done:
                    break
            print('[INFO] Episode {}/{}| Reward {}'.format(episode + 1, n_episodes, episode_reward))


        