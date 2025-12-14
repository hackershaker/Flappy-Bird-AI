import time
import numpy as np
from agents.q_learning_agent import QLearningAgent
from flappy_bird_env import FlappyBirdEnv


class QLearningPlay:
    def __init__(self, q_table_path="best_q_table.npy", bins=(10, 10, 10, 10)):
        self.env = FlappyBirdEnv(render=True)
        self.agent = QLearningAgent(self.env, bins=bins)
        self.agent.q_table = np.load(q_table_path)

        # 완전 greedy하게 진행
        self.agent.epsilon = 0.0

    def run(self, delay=0.015):
        state = self.env.reset()
        total_reward = 0
        done = False

        print("Q-Learning 플레이 시작")

        while not done:
            action = self.agent.act(state)
            next_state, reward, done = self.env.step(action)
            state = next_state
            total_reward += reward

            self.env.render()
            time.sleep(delay)

        print(f"Game Over! Total Reward: {total_reward}")
