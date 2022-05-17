from asappo import ASAPPO
import pogema
from pogema.grid import GridConfig
import gym
from appo import APPOHolder
import numpy as np
from time import time
import torch
import random
torch.manual_seed(13)
random.seed(13)
np.random.seed(13)

if __name__ == '__main__':

    grid_config = GridConfig(num_agents=32,  # number of agents
                         size=16, # size of the grid
                         density=0.4,  # obstacle density
                         seed=11,  # set to None for random 
                                  # obstacles, agents and targets 
                                  # positions at each reset
                         max_episode_steps=32,  # horizon
                         obs_radius=5,  # defines field of view
                         )

    env = gym.make("Pogema-v0", grid_config=grid_config, integration="SampleFactory")
    

    obs = env.reset()
    env.render()
    done = [False, ...]
    info = {}
    reward = [0, ...]

    asappo = ASAPPO('weights/c164')
    appo = APPOHolder('weights/c164')

    frame = 0
    start = time()
    while not all(done):
        obs, reward, done, info = asappo.act(env, obs)
        #env.render()
        frame += 1
    
    print(frame/(time() - start), frame)
    print(info[0]['episode_extra_stats']['CSR'])
    print(np.mean([x['episode_extra_stats']['ISR'] for x in info]))

    env = gym.make("Pogema-v0", grid_config=grid_config, integration="SampleFactory")
    

    obs = env.reset()
    done = [False, ...]
    info = {}
    reward = [0, ...]

    appo = APPOHolder('weights/c164')

    frame = 0
    start = time()
    while not all(done):
        actions = appo.act(obs)
        obs, reward, done, info = env.step(actions)
        frame += 1

    print(frame/(time() - start), frame)
    print(info[0]['episode_extra_stats']['CSR'])
    print(np.mean([x['episode_extra_stats']['ISR'] for x in info]))