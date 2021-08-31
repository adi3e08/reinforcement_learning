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
               torch.tensor(A, dtype=torch.float, device=self.device),\
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
        self.fc2 = torch.nn.Linear(64+action_size, 64)
        self.fc3 = torch.nn.Linear(64, 1)

    def forward(self, x, a):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(torch.cat((y1,a),1)))
        y = self.fc3(y2).view(-1)        
        return y

# Fully Connected Policy network
class mu_FC(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(mu_FC, self).__init__()
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, action_size)

    def forward(self, x):
        y1 = F.relu(self.fc1(x))
        y2 = F.relu(self.fc2(y1))
        y = F.tanh(self.fc3(y2))        
        return y

# DDPG
class DDPG:
    def __init__(self, arglist, env):
        self.arglist = arglist
        self.env = env
        self.device = torch.device("cpu")
        # self.device = torch.device("cuda:0")
        self.actor = mu_FC(self.env.state_size,self.env.action_size).to(self.device)
        self.actor_target = mu_FC(self.env.state_size,self.env.action_size).to(self.device)       
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Q_FC(self.env.state_size,self.env.action_size).to(self.device)
        self.critic_target = Q_FC(self.env.state_size,self.env.action_size).to(self.device)       
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.replay_buffer = ReplayBuffer(self.arglist.replay_size, self.device)
        self.actor_loss_fn =  torch.nn.MSELoss()
        self.critic_loss_fn =  torch.nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.arglist.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.arglist.lr)
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
        checkpoint = {'actor' : self.actor.state_dict()}
        torch.save(checkpoint, os.path.join(self.model_dir, name))

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])

    def sample(self, o, noise_scale):
        with torch.set_grad_enabled(False):
            a = self.actor(torch.tensor(o, dtype=torch.float, device=self.device).unsqueeze(0)).clone().detach().cpu().numpy()[0]
        a = (a+1)*(self.env.action_high-self.env.action_low)/2+self.env.action_low
        a += noise_scale * np.random.randn(self.env.action_size)
        return a

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def train(self):
        writer = SummaryWriter(log_dir=self.tensorboard_dir)

        for episode in range(self.arglist.episodes):
            o = self.env.reset()
            ep_r = 0
            while True:
                a = self.sample(o, (self.env.action_high-self.env.action_low)/10)  
                o_1, r, done, info = self.env.step(a)
                self.replay_buffer.push(o, a, r, o_1, int(done))
                ep_r += r
                o = o_1
                if self.replay_buffer.__len__() < arglist.replay_fill:
                    pass
                else :
                    O, A, R, O_1, D = self.replay_buffer.sample(self.arglist.batch_size)

                    q_value = self.critic(O, A)

                    with torch.set_grad_enabled(False):
                        next_q_value = self.critic_target(O_1,self.actor_target(O_1))                                    
                    expected_q_value = R + self.arglist.gamma * next_q_value * (1 - D)

                    critic_loss = self.critic_loss_fn(q_value, expected_q_value)
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    for param in self.critic.parameters():
                        param.requires_grad = False

                    actor_loss = - torch.mean(self.critic(O,self.actor(O)))
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    for param in self.critic.parameters():
                        param.requires_grad = True

                    self.soft_update(self.actor_target, self.actor, self.arglist.tau)
                    self.soft_update(self.critic_target, self.critic, self.arglist.tau)

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
                a = self.sample(o, 0.0)   
                o_1, r, done, info = self.env.step(a)
                ep_r += r
                o = o_1
                if done:
                    ep_r_list.append(ep_r)
                    break
        return ep_r_list    

def parse_args():
    parser = argparse.ArgumentParser("DQN")
    parser.add_argument("--exp-name", type=str, default="expt_pendulum", help="name of experiment")
    parser.add_argument("--episodes", type=int, default=6000, help="number of episodes")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=2.5e-4, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--tau", type=float, default=0.01, help="soft target update parameter")
    parser.add_argument("--replay-size", type=int, default=100000, help="replay buffer size")
    parser.add_argument("--replay-fill", type=int, default=10000, help="elements in replay buffer before training starts")
    parser.add_argument("--eval-every", type=int, default=500, help="eval every _ episodes")
    parser.add_argument("--eval-over", type=int, default=100, help="eval over _ episodes")
    return parser.parse_args()

def make_env():
    env = gym.make('Pendulum-v0')
    env.state_size = 3
    env.action_size = 1
    env.action_low = -2
    env.action_high = 2
    # env = gym.make('MountainCarContinuous-v0')
    # env.state_size = 2
    # env.action_size = 1
    # env.action_low = -1
    # env.action_high = 1
    # env = gym.make('BipedalWalker-v3')
    # env.state_size = 24
    # env.action_size = 4
    # env.action_low = -1
    # env.action_high = 1
    return env

if __name__ == '__main__':
    arglist = parse_args()
    env = make_env()
    ddpg = DDPG(arglist, env)
    ddpg.train()

