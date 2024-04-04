# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 19:39:53 2024

@author: tanbi
"""
import gym
from gym import spaces
import pygame
import random
import numpy as np
from collections import deque
import tensorflow as tf


class PongEnv(gym.Env):
    def __init__(self):
        super(PongEnv, self).__init__()
        
        # Initialize pygame
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Pong')
        
        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # Actions: 0 (move up) or 1 (move down)
        self.observation_space = spaces.Box(low=0, high=255, shape=(600, 800, 3), dtype=np.uint8)  # RGB pixel values
        
        # Define game variables
        self.bg_color = pygame.Color('grey12')
        self.light_grey = (200, 200, 200)
        self.ball_speed_x = 7 * random.choice((1,-1))
        self.ball_speed_y = 7 * random.choice((1,-1))
        self.player_speed = 0
        self.opponent_speed = 7
        self.player_score = 0
        self.opponent_score = 0
        self.game_font = pygame.font.Font("freesansbold.ttf", 32)
        self.score_time = None
        self.ball = pygame.Rect(self.screen_width/2 - 15, self.screen_height/2 - 15, 30, 30)
        self.player = pygame.Rect(self.screen_width - 20, self.screen_height/2 - 70, 10, 140)
        self.opponent = pygame.Rect(10, self.screen_height/2 - 70, 10, 140)

    def step(self, action):
        # Take action and return next observation, reward, done flag, and info
        self._handle_player_action(action)
        self._handle_opponent_action()
        self._handle_ball_movement()
        observation = self._get_observation()
        reward = self._get_reward()
        done = self._check_done()
        info = {}
        return observation, reward, done, info

    def reset(self):
        # Reset environment to initial state and return initial observation
        self.ball_speed_x = 7 * random.choice((1,-1))
        self.ball_speed_y = 7 * random.choice((1,-1))
        self.player_speed = 0
        self.player_score = 0
        self.opponent_score = 0
        self.score_time = None
        self.ball = pygame.Rect(self.screen_width/2 - 15, self.screen_height/2 - 15, 30, 30)
        self.player = pygame.Rect(self.screen_width - 20, self.screen_height/2 - 70, 10, 140)
        self.opponent = pygame.Rect(10, self.screen_height/2 - 70, 10, 140)
        return self._get_observation()

    def render(self):
        # Render environment (optional)
        
        pygame.display.update()
        self.clock.tick(60)
        pass

    def close(self):
        # Close environment (optional)
        pygame.quit()
        
    def _handle_player_action(self, action):
        # Handle player action
        if action == 0:
            self.player_speed -= 7
        elif action == 1:
            self.player_speed += 7

    def _handle_opponent_action(self):
        # Handle opponent action
        if self.opponent.top < self.ball.y:
            self.opponent.top += self.opponent_speed
        elif self.opponent.bottom > self.ball.y:
            self.opponent.bottom -= self.opponent_speed
        if self.opponent.top <= 0:
            self.opponent.top = 0
        if self.opponent.bottom >= self.screen_height:
            self.opponent.bottom = self.screen_height
    
    def _handle_ball_movement(self):
        # Handle ball movement
        self.ball.x += self.ball_speed_x
        self.ball.y += self.ball_speed_y
        if self.ball.top <= 0 or self.ball.bottom >= self.screen_height:
            self.ball_speed_y *= -1
        if self.ball.left <= 0:
            self.player_score += 1
            self.score_time = pygame.time.get_ticks()
        if self.ball.right >= self.screen_width:
            self.opponent_score += 1
            self.score_time = pygame.time.get_ticks()
        if self.ball.colliderect(self.player) or self.ball.colliderect(self.opponent):
            self.ball_speed_x *= -1
            
    def _get_observation(self):
        # Get observation (current screen state)
        screen_state = pygame.surfarray.array3d(pygame.display.get_surface())
        surface = pygame.surfarray.make_surface(screen_state)
        resized_surface = pygame.transform.scale(surface, (800, 600))
        resized_state = pygame.surfarray.array3d(resized_surface)
        return resized_state
        
        
    def _get_reward(self):
        # Get reward
        return self.player_score - self.opponent_score
    
    def _check_done(self):
        # Check if episode is done
        return self.player_score == 5 or self.opponent_score == 5

# Register custom environment with Gym


env = PongEnv()

class DQNAgent:
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, learning_rate=0.01, batch_size=32, memory_size=10000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.q_network = self._build_q_network()
        self.target_network = self._build_q_network()
        self.target_network.set_weights(self.q_network.get_weights())
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
    def _build_q_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=self.state_dim),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')
        ])
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        else:
            q_values = self.q_network.predict(state.reshape(1, *self.state_dim))[0]
            return np.argmax(q_values)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q_values = self.target_network.predict(next_state.reshape(1, *self.state_dim))[0]
                target = reward + self.gamma * np.amax(next_q_values)
            q_values = self.q_network.predict(state.reshape(1, *self.state_dim))[0]
            q_values[action] = target
            states.append(state)
            targets.append(q_values)
        
        states = np.array(states)
        targets = np.array(targets)
        self.q_network.fit(states, targets, epochs=1, verbose=0)
        
    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def compile_model(self):
        self.q_network.compile(optimizer=self.optimizer, loss='mse')
    
    def interact_with_environment(self, env, episodes=3):
        self.compile_model()
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                env.render()
                
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                self.replay()
                self.update_target_network()
                self.decay_epsilon()
            print("Episode:", episode + 1, "Total Reward:", total_reward)
        env.close()
    
    def test(self, env, episodes=3):
        total_rewards = []
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                action = np.argmax(self.q_network.predict(state.reshape(1, *self.state_dim))[0])
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
            total_rewards.append(total_reward)
            print("Episode:", episode + 1, "Total Reward:", total_reward)
        avg_reward = sum(total_rewards) / len(total_rewards)
        print("Average Reward:", avg_reward)


state_dim = env.observation_space.shape
action_dim = env.action_space.n

agent = DQNAgent(state_dim, action_dim)
agent.interact_with_environment(env)
#agent.test(env)


