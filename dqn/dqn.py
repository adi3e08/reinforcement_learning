import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import gym
from collections import deque
import random

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.buffer = deque(maxlen=capacity)
        self.device = device
    
    def push(self, o, a, r, o_1, d):            
        self.buffer.append((o, a, r, o_1, d))
    
    def sample(self, batch_size):
        O, A, R, O_1, D = zip(*random.sample(self.buffer, batch_size))
        return torch.tensor(O, dtype=torch.float, device=self.device),\
               torch.tensor(A, dtype=torch.long, device=self.device),\
               torch.tensor(R, dtype=torch.float, device=self.device),\
               torch.tensor(O_1, dtype=torch.float, device=self.device),\
               torch.tensor(D, dtype=torch.float, device=self.device)

    def __len__(self):
        return len(self.buffer)

# Fully Connected Q network
class Q_FC(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Q_FC, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, action_size)

    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        y = self.fc3(y2)        
        return y

# DQN
class DQN:
    def __init__(self, arglist, env):
        self.arglist = arglist
        self.env = env
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda:0")
        self.Q = Q_FC(self.env.state_size,self.env.action_size).to(self.device)
        self.Q_target = Q_FC(self.env.state_size,self.env.action_size).to(self.device)       
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)
        self.loss_fn =  torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.arglist.lr)
        self.exp_dir = os.path.join("./log", self.arglist.exp_name)
        self.model_dir = os.path.join(self.exp_dir, "models")
        self.tensorboard_dir = os.path.join(self.exp_dir, "tensorboard")
        if os.path.exists("./log"):
            pass            
        else:
            os.mkdir("./log")
        os.mkdir(self.exp_dir)
        os.mkdir(os.path.join(self.tensorboard_dir))
        os.mkdir(self.model_dir)

    def save_checkpoint(self, name):
        checkpoint = {'Q' : self.Q.state_dict()}
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.Q.load_state_dict(checkpoint['Q'])

    def sample(self, o, epsilon=0.0):
        with torch.set_grad_enabled(False):
            a_best = self.Q(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0)).max(1)[1][0].item()
        a_rand = np.random.randint(0,self.env.action_size)
        a = np.random.choice(np.array([a_rand, a_best]), p = [epsilon, 1.0 - epsilon])
        return a

    def train(self):
        epsilon = 1.0
        update_count = 0
        target_update_count = arglist.target_update_every-1
        writer = SummaryWriter(log_dir=self.tensorboard_dir)

        for episode in range(self.arglist.episodes):
            o = self.env.reset()
            ep_r = 0
            while True:
                a = self.sample(o, epsilon)  
                o_1, r, done, info = self.env.step(a)
                self.replay_buffer.push(o, a, r, o_1, int(done))
                ep_r += r
                o = o_1
                epsilon = max(self.arglist.min_epsilon, epsilon-self.arglist.epsilon_decay)
                if self.replay_buffer.__len__() < arglist.replay_fill:
                    pass
                else :
                    if update_count == 0:
                        O, A, R, O_1, D = self.replay_buffer.sample(self.arglist.batch_size)

                        q_values = self.Q(O)                                                
                        q_value = q_values.gather(1, A.unsqueeze(1)).squeeze(1)

                        with torch.set_grad_enabled(False):
                            next_q_values = self.Q_target(O_1)
                        next_q_value = next_q_values.max(1)[0]                                        
                        expected_q_value = R + self.arglist.gamma * next_q_value * (1 - D)

                        loss = self.loss_fn(q_value, expected_q_value)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        target_update_count = (target_update_count+1)%arglist.target_update_every

                    update_count = (update_count+1)%arglist.update_every 
                    if target_update_count == 0:
                        self.Q_target.load_state_dict(self.Q.state_dict())  

                if done:
                    writer.add_scalar('ep_r', ep_r, episode)
                    if episode % self.arglist.eval_every == 0 or episode == self.arglist.episodes-1:
                        eval_ep_r_list = self.eval(self.arglist.eval_over)
                        writer.add_scalar('eval_ep_r', np.mean(eval_ep_r_list), episode)
                        self.save_checkpoint(str(episode)+".ckpt")
                    break   

    def eval(self, episodes):
        ep_r_list = []
        for episode in range(episodes):
            o = self.env.reset()
            ep_r = 0
            while True:
                a = self.sample(o)   
                o_1, r, done, info = self.env.step(a)
                ep_r += r
                o = o_1
                if done:
                    ep_r_list.append(ep_r)
                    break
        return ep_r_list    

def parse_args():
    parser = argparse.ArgumentParser("DQN")
    parser.add_argument("--exp-name", type=str, default="expt_1", help="name of experiment")
    parser.add_argument("--episodes", type=int, default=25000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("--min-epsilon", type=float, default=0.1, help="min epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=1e-5, help="reduce epsilon by _ every step")
    parser.add_argument("--update-every", type=int, default=4, help="train after every _ steps")
    parser.add_argument("--target-update-every", type=int, default=100, help="update target every _ updates")
    parser.add_argument("--replay-size", type=int, default=10000, help="replay buffer size")
    parser.add_argument("--replay-fill", type=int, default=1000, help="elements in replay buffer before training starts")
    parser.add_argument("--eval-every", type=int, default=1000, help="eval every _ episodes")
    parser.add_argument("--eval-over", type=int, default=100, help="eval over _ episodes")
    return parser.parse_args()

def make_env():
    # env = gym.make('CartPole-v1')
    # env.state_size = 4
    # env.action_size = 2
    env = gym.make('MountainCar-v0')
    env.state_size = 2
    env.action_size = 3
    return env

if __name__ == '__main__':
    arglist = parse_args()
    env = make_env()
    dqn = DQN(arglist, env)
    dqn.train()