from asappo import ASAPPO
import pogema
from pogema.grid import GridConfig
import gym
from appo import APPOHolder
import numpy as np
from time import time
import torch
import random
import json
torch.manual_seed(13)
random.seed(13)
np.random.seed(13)

if __name__ == '__main__':

    configs = [
        GridConfig(num_agents=24,
                         size=12,
                         density=0.3,
                         seed=i, 
                         max_episode_steps=32,
                         obs_radius=5,
                         )
        for i in range(50)
    ]
    results =  {'ASAPPO': [], 'APPO': []}
    for grid_config in configs:

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
            obs, reward, done, info = asappo.act(env, obs, 32)
            #env.render()
            frame += 1

        results['ASAPPO'].append({'CSR': info[0]['episode_extra_stats']['CSR'], 
                                  'ISR': np.mean([x['episode_extra_stats']['ISR'] for x in info]),
                                  'makespan': frame, 'FPS': frame/(time() - start)})

        env = gym.make("Pogema-v0", grid_config=grid_config, integration="SampleFactory")
        

        observations = env.reset()
        done = [False, ...]
        info = {}
        reward = [0, ...]

        frame = 0
        start = time()
        while not all(done):
            if observations[0].shape[-1] != 11:
                init_dim = observations[0].shape[1]
                obs_ = [np.zeros((observations[0].shape[0], 11, 11)) for _ in range(len(observations))]
                for i in range(len(obs_)):
                    obs_[i][:, :init_dim,:init_dim] = observations[i]
                observations = obs_
            actions = appo.act(observations)
            observations, reward, done, info = env.step(actions)
            frame += 1

        results['APPO'].append({'CSR': info[0]['episode_extra_stats']['CSR'], 
                                  'ISR': np.mean([x['episode_extra_stats']['ISR'] for x in info]),
                                  'makespan': frame, 'FPS': frame/(time() - start)})
    #print(results)
    results['ASAPPO'] = {'CSR': np.mean([x['CSR'] for x in results['ASAPPO']]), 
                         'ISR': np.mean([x['ISR'] for x in results['ASAPPO']]),
                         'makespan': np.mean([x['makespan'] for x in results['ASAPPO']]),
                         'FPS': np.mean([x['FPS'] for x in results['ASAPPO']])}
    results['APPO'] = {'CSR': np.mean([x['CSR'] for x in results['APPO']]), 
                         'ISR': np.mean([x['ISR'] for x in results['APPO']]),
                         'makespan': np.mean([x['makespan'] for x in results['APPO']]),
                         'FPS': np.mean([x['FPS'] for x in results['APPO']])}
    print(results)
    with open('result.json', 'w') as fout:
        json.dump(results, fout)