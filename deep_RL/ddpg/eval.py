import numpy as np
import torch
import torch.nn.functional as F
import gym

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

def main():
    env = make_env()
    device = torch.device("cpu")
    #device = torch.device("cuda:0")
    actor = mu_FC(env.state_size,env.action_size).to(device)
    checkpoint = torch.load("./log/expt_pendulum/models/4500.ckpt")
    actor.load_state_dict(checkpoint['actor'])

    no_episodes = 10
    for episode in range(no_episodes):
        o_t = env.reset()
        ep_r = 0
        while True:
            env.render()
            with torch.set_grad_enabled(False):
                a_t = actor(torch.tensor(o_t, dtype=torch.float, device=device).unsqueeze(0)).clone().detach().cpu().numpy()[0]
            a_t = (a_t+1)*(env.action_high-env.action_low)/2+env.action_low  
            o_t_1, r_t, done, info = env.step(a_t)
            ep_r += r_t
            o_t = o_t_1
            if done:
                print("Episode finished with total reward ",ep_r)
                break
    env.close()

if __name__ == '__main__':
    main()