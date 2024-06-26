# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:38:37 2024

@author: tanbi
"""
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

def plotlearning(x, scores, epsilons, filename, window, lines = None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)
    
    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")
    
    N=len(scores)
    running_avg=np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window) : (t+1)])
        
    ax2.scatter(x, running_avg, color="C1")
    #ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    #ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    #ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    #ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")
    
    if lines is not None:
        for line in lines:
            plt.axvline(x=line)
            
    plt.savefig(filename)
    
def plotlearningNoEpsilons(scores, filename, x=None, window=5):
    N=len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x=[i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg)
    plt.savefig(filename)

class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip
        
    def Step(self, action):
        t_reward = 0.0
        done = False
        for _ in range (self._skip):
            obs = self.env.step(action)[0] 
            reward = self.env.step(action)[1]
            done = self.env.step(action)[2]
            info = self.env.step(action)[3]
            
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info
    
class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low =0, high = 255,
                                                shape = (80,80,4), dtype = np.uint8)
        
    def observation(self, obs):
        return PreProcessFrame.process(obs)
    
    @staticmethod
    def process(frame):
        new_frame = np.reshape(frame, frame.shape).astype(np.float32)
        new_frame = 0.299*new_frame[:,:,0] + 0.587*new_frame[:,:,1] + \
                    0.114*new_frame[:,:,2]
 
        new_frame = new_frame[35:195:2, ::2].reshape(80,80,1)
        
        return new_frame.astype(np.uint8)
    
    
class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box( low = 0.0, high=1.0,
                                shape =(self.observation_space.shape[-1],
                                        self.observation_space.shape[0],
                                        self.observation_space.shape[1]),
                                dtype=np.float32)
    
    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)
    
class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0
    
class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                                env.observation_space.low.repeat(n_steps, axis=0),
                                env.observation_space.high.repeat(n_steps, axis=0),
                                dtype=np.float32)
    def Reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())
    
    def observation(self, observation):
        
        frame = observation[0]
        
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = frame
        return self.buffer
    
def make_env(env_name):
    env = gym.make(env_name)
    env=SkipEnv(env)
    env=PreProcessFrame(env)
    env=MoveImgChannel(env)
    env=BufferWrapper(env, 4)
    return ScaleFrame(env)        
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    