import gym as gym
import torch
from torch.optim import Adam
import numpy as np
import ReinforcmentLearning.Reinforce

gamma = 0.99
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(pi, optimizer):
    T = len(pi.rewards)
    rets = np.empty(T, dtype=np.float32)
    future_ret = 0.0

    for t in reversed(range(T)):
        future_ret = pi.rewards[t] + gamma * future_ret
        rets[t] = future_ret

    rets = torch.tensor(rets)
    log_probs = torch.stack(pi.log_probs)
    loss = -log_probs * rets  # gradient term
    loss = torch.sum(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def main():
    env = gym.make('CartPole-v0')
    in_dim = env.observation_space.shape[0]  # 4
    out_dim = env.action_space.n  # 2
    pi = Pi(in_dim, out_dim)

    optimizer = Adam(pi.parameters(), lr=0.01)
    for epi in range(200):
        state = env.reset()
        for t in range(1000):
            action = pi.act(state)
            state, reward, done, _ = env.step(action)
            pi.rewards.append(reward)
            env.render()
            if done:
                break
        loss = train(pi, optimizer)
        total_reward = sum(pi.rewards)
        solved = total_reward > 195.0
        pi.onpolicy_reset()
        print(f'Episode {epi}, loss {loss} total reward ={total_reward}, solved = {solved}')


if __name__ == '__main__':
    main()
