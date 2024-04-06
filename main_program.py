# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:26:43 2024

@author: tanbi
"""

import numpy as np
from dqn_keras import Agent
from utils import plotlearning, make_env

if __name__== '__main__':
    env = make_env('PongNoFrameskip-v4')
    num_games = 250
    load_checkpoint = False
    best_score = -21
    
    agent = Agent(gamma=0.99, epsilon=1.0, alpha=0.0001, input_dims=(4,80,80),
                  n_actions=6, mem_size=25000, eps_min=0.02, batch_size=32,
                  replace=1000, eps_dec=1e-5)
    
    if load_checkpoint:
        agent.load_models()
        
    filename = 'PongNoFrameSkip -v4.png'
    
    score, eps_history = [], []
    n_steps = 0;
    
    for i in range(num_games):
        score = 0
        observation = env.Reset()
        done = False
        while not done :
            action = agent.choose_action(observation)
            #print(env.step(action).shape)
            #observation_ = env.step(action)[0]
            #reward = env.step(action)[1]
            #done = env.step(action)[2]
            #truncated = env.step(action)[3]
            #info = env.step(action)[4]            
            step_info = env.step(action)
            if len(step_info) == 4:
                observation, reward, done, info = step_info
            elif len(step_info) == 5:
                observation, reward, done, truncated, info = step_info
            n_steps += 1
            score += reward
            if not load_checkpoint:
                agent.store_transition(observation, action,
                                       reward, observation_, int(done))
                agent.learn()
                
            else:
                env.render()
                
            observation = observation_
        score.append(score)
        
        avg_score = np.mean(score[-100:1])
        print('episode',i, 'score', score, 'average score % .2f' % avg_score, 'epsilon %.2f' % agent.epsilon,
              'steps', n_steps)
        
        if avg_score > best_score:
            agent.save_models()
            print('avg score %.2f better thsn best score %.2f' %
                  (avg_score, best_score))
            best_score = avg_score
            
        eps_history.append(agent.epsilon)
        
    x = [i + 1 for i in range(num_games)]
    plotlearning(x, score, eps_history, filename)
        
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                