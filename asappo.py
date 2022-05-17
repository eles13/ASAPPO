import numpy as np
import gym
from pogema.grid_config import GridConfig
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.animation import AnimationMonitor
from appo import APPOHolder
from astar import coop_astar, manhattan_distance, SearchTree
import random

class ASAPPO:
    
    def __init__(self, weights_path):
        self.appo = APPOHolder(path=weights_path)
        
    def act(self, env, observations):
        positions = env.get_agents_xy()
        targets = env.get_targets_xy()
        conflicting = self._find_conflicts(positions, targets, env.grid.config.obs_radius)
        actions_for_confl = coop_astar(conflicting, env.grid.config.num_agents, env.grid, manhattan_distance, SearchTree)
        if observations[0].shape[-1] != 11:
            init_dim = observations[0].shape[1]
            obs_ = [np.zeros((observations[0].shape[0], 11, 11)) for _ in range(len(observations))]
            for i in range(len(obs_)):
                obs_[i][:, :init_dim,:init_dim] = observations[i]
            observations = obs_
        actions = self.appo.act(observations)
        #print(actions_for_confl)
        for ag in actions_for_confl:
            actions[ag] = actions_for_confl[ag]
        observations, rewards, dones, infos = env.step(actions)
        return observations, rewards, dones, infos

    def _find_conflicts(self, positions, targets, obs_radius):
        conflicting = set()
        for i in range(len(positions)):
            if positions[i] != targets[i]:
                for j in range(len(positions)):
                    if i != j and abs(positions[i][0] - positions[j][0]) <= obs_radius and \
                        abs(positions[i][0] - positions[j][0]) <= obs_radius and \
                            abs(positions[i][1] - positions[j][1]) <= obs_radius and \
                                positions[j] != targets[j]:
                        conflicting.add(i)
                        conflicting.add(j)
        return list(conflicting)