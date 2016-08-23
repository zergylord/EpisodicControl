import gym
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as misc
import time
from sklearn.neighbors import NearestNeighbors
from ops import *
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
class EpisodicControlAgent(object):
    def __init__(self,num_actions):
        self.test_length = 0

        self.gamma = 1.0
        self.epsilon = .005
        self.num_neighbors = 5
        self.n_act = num_actions
        print(self.n_act)
        self.knn = []
        for a in range(self.n_act):
            self.knn.append(NearestNeighbors(n_neighbors=self.num_neighbors,metric='euclidean'))#,algorithm='brute'))
        self.s_dim = 84*84 #env.observation_space.shape[0]
        self.rep_dim = 64
        self.M = np.random.randn(self.s_dim,self.rep_dim)
        self.mem_size = int(1e6)
        self.S = np.zeros((self.n_act,self.mem_size,self.rep_dim))
        self.R = np.zeros((self.n_act,self.mem_size,))
        self.last_used = np.tile(np.asarray(range(-self.mem_size,0)),[self.n_act,1])
        self.mem_ind = np.zeros((self.n_act,),dtype=int)
        self.warming = True
        self.steps = 0
        self.total_hits = 0
        self.cumr = 0.0
    def _process_obs(self,obs):
        s = np.float32(obs)/255.0
        s = np.reshape(s,self.s_dim)
        return np.matmul(s,self.M)
    def start_episode(self, observation):
        self.last_action = np.random.randint(self.n_act)
        self.last_state = self._process_obs(observation)
        self.last_match = False
        self.episode_rewards = []
        self.episode_actions = []
        self.episode_states = []
        self.episode_matches = []
        self.episode_match_inds = []
        return self.last_action
    def _get_action(self,rep):
        #global knn,n_act
        max_q_val = float("-inf")
        act = 0
        tied = []
        nearby = []
        exact_match = {}
        for a in range(self.n_act):
            nearby.append([])
            dists,inds = self.knn[a].kneighbors(np.expand_dims(rep,0))
            dists = dists[0]
            inds = inds[0]
            nearby[a] = inds
            if 0.0 in dists:
                hit_ind = inds[dists == 0.0][0]
                #print('hit!',hit_ind,R[a][hit_ind])
                q_val = self.R[a][hit_ind]
                exact_match[a] = hit_ind
            else:
                q_val = np.mean(self.R[a][inds])
            if q_val > max_q_val:
                max_q_val = q_val
                act = a
                tied = []
                a_not_in_tied = True
            elif a != act and q_val == max_q_val:
                if not tied:
                    tied.append(act)
                tied.append(a)
                act = a
        if tied:
            act = np.random.choice(tied) 
        return act,nearby,exact_match
    def step(self,reward,observation):
        self.cumr+=reward
        self.steps+=1
        self.episode_rewards.append(np.clip(reward,-1,1))
        self.episode_actions.append(self.last_action)
        self.episode_states.append(self.last_state)
        self.episode_matches.append(self.last_match)
        if self.last_match:
            self.episode_match_inds.append(self.last_match_ind)

        state = self._process_obs(observation)
        #----action selection--------------
        if not self.warming:
            action,nearby,match = self._get_action(state)
            for a in range(self.n_act):
                self.last_used[a,nearby[a]] = self.steps

        if np.random.rand() < self.epsilon or self.warming:
            action = np.random.randint(self.n_act)
        if not self.warming and action in match:
            self.last_match = True
            self.last_match_ind = match[action]
            self.total_hits+=1
        else:
            self.last_match = False
        self.last_action = action
        self.last_state = state
        return action

    def end_episode(self,reward,terminal=True):
        self.cumr+=reward
        self.episode_rewards.append(np.clip(reward,-1,1))
        self.episode_actions.append(self.last_action)
        self.episode_states.append(self.last_state)
        self.episode_matches.append(self.last_match)
        if self.last_match:
            self.episode_match_inds.append(self.last_match_ind)

        if self.warming:
            if self.steps > 250:
                self.warming = False
        self.episode_rets = np.asarray(compute_return(self.episode_rewards,self.gamma))
        self.episode_states = np.asarray(self.episode_states)
        self.episode_actions = np.asarray(self.episode_actions)
        self.episode_matches = np.asarray(self.episode_matches)
        if np.any(self.episode_matches):
            #update matched return estimates
            match_act_inds = self.episode_actions[self.episode_matches]
            self.R[match_act_inds,self.episode_match_inds] = np.maximum(self.R[match_act_inds,self.episode_match_inds]
                    ,self.episode_rets[self.episode_matches])
            if not np.all(self.episode_matches):
                #remove matches from list to add to memory
                neg = np.logical_not(self.episode_matches)
                self.episode_rets = self.episode_rets[neg]
                self.episode_actions = self.episode_actions[neg]
                self.episode_states = self.episode_states[neg]
                add_memories = True
            else:
                add_memories = False
        else:
            add_memories = len(self.episode_states)>0
        if add_memories:
            #-----add stuff to memory------------
            episode_reps = np.asarray(self.episode_states)
            for a in range(self.n_act):
                mask = self.episode_actions==a
                n_reps = len(self.episode_actions[mask])
                if n_reps > 0:
                    replace_these = np.argpartition(self.last_used[a],n_reps-1)[:n_reps]
                    self.last_used[a][replace_these] = self.steps
                    self.S[a,replace_these] = episode_reps[mask]
                    self.R[a,replace_these] = self.episode_rets[mask]
                    if self.mem_ind[a] + n_reps < self.mem_size:
                        self.mem_ind[a] += n_reps
                    else:
                        self.mem_ind[a] = self.mem_size
                    self.knn[a].fit(self.S[a][:self.mem_ind[a]])
    def finish_epoch(self,epoch):
        print(epoch,self.cumr,self.total_hits/self.steps)
        self.cumr = 0.0
        #self.total_hits = 0
