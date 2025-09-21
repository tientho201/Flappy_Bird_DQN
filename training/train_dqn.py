import os 
import yaml
import argparse
import numpy as np 
from collections import deque
from torch.utils.tensorboard import SummaryWriter

import gym 
from envs.flappy_env import FlappyBirdEnv
from agents.dqn import DQNAgent
from training.replay_buffer import ReplayBuffer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/dqn.yaml")
    return p.parse_args()

def train(config):
    env = FlappyBirdEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim , lr = config["learning_rate"], gamma = config["gamma"], epsilon = config["epsilon_start"], eps_min = config["epsilon_end"], eps_decay = config["epsilon_decay"])
    replay_buffer = ReplayBuffer(capacity = config["replay_buffer_size"])
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    epsilon = config["epsilon_start"]
    epsilon_decay = (config['epsilon_start'] - config['epsilon_end']) / config['epsilon_decay']
    total_steps = 0
    best_score = 0
    for episode in range(1, config["max_episode"] + 1):
        state = env.reset()
        ep_reward = 0
        for t in range(config["max_step_per_episode"]):
            action = agent.act(state,epsilon)
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            total_steps += 1 
            if done:
                break
            if epsilon > config["epsilon_end"]:
                epsilon -= epsilon_decay
            if total_steps % config["target_update"] == 0:
                agent.update_target()
            if len(replay_buffer) >= config["batch_size"]:
                loss = agent.update(replay_buffer, config["batch_size"])
                if total_steps % 100 == 0:
                    print(f"Step {total_steps}, Loss: {loss:.4f}")
        print(f"Episode {episode}, Reward: {ep_reward}, Epsilon: {epsilon:.2f}")
        # checkpoint
        if info.get('score', 0) > best_score:
            best_score = info.get('score', 0)
            os.makedirs('data/checkpoints', exist_ok=True)
            agent.save('data/checkpoints/dqn_best.pth')
        writer.add_scalar("Reward/episode", ep_reward, episode)
        writer.add_scalar("Epsilon/value", epsilon, episode)



    env.close()

if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    train(config)