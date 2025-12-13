"""
Q-러닝 에이전트 구현
- 상태를 이산화(discretize) 후 Q-테이블 학습
- 연속 상태 -> Q-테이블 인덱스로 변환
"""

import numpy as np

from agents.base_agent import BaseAgent


'''

def discretize_state(state, bins=(10, 10, 10, 10)):
    """
    연속 상태를 이산 상태로 변환
    :param state: [bird_y, bird_vel, pipe_x, pipe_y]
    :param bins: 각 상태를 나눌 구간 수
    :return: 이산 상태 튜플
    """
    bird_y, bird_vel, pipe_x, pipe_y = state
    y_bin = min(int(bird_y / (512 / bins[0])), bins[0] - 1)
    vel_bin = min(int((bird_vel + 20) / (40 / bins[1])), bins[1] - 1)
    pipe_x_bin = min(int(pipe_x / (288 / bins[2])), bins[2] - 1)
    pipe_y_bin = min(int(pipe_y / (512 / bins[3])), bins[3] - 1)
    return (y_bin, vel_bin, pipe_x_bin, pipe_y_bin)

'''


class QLearningAgent(BaseAgent):
    def __init__(self, env, bins=(10, 10, 10, 10), alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.bins = bins
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q테이블 = 상태 별 구간 수 X 행동 수, 초기엔 0으로 초기화
        self.q_table = np.zeros(bins + (2,))

    def discretize_state(self, state):
        bird_y, bird_vel, pipe_x, pipe_y = state
        y_bin = min(int(bird_y / (512 / self.bins[0])), self.bins[0] - 1)
        vel_bin = min(int((bird_vel + 20) / (40 / self.bins[1])), self.bins[1] - 1)
        pipe_x_bin = min(int(pipe_x / (288 / self.bins[2])), self.bins[2] - 1)
        pipe_y_bin = min(int(pipe_y / (512 / self.bins[3])), self.bins[3] - 1)
        return (y_bin, vel_bin, pipe_x_bin, pipe_y_bin)

    def act(self, state):
        d_state = self.discretize_state(state)  # 상태 이산화
        # 만약 ε 값보다 작다면 탐험
        if np.random.rand() < self.epsilon:
            return np.random.randint(2)  # 행동 0,1 중 하나를 선택
        # 아니라면 d_state에서 가능한 행동 중 Q값이 가장 높은 행동 선택
        return np.argmax(self.q_table[d_state])

    def act_eval(self, state):
        d_state = self.discretize_state(state)
        return int(np.argmax(self.q_table[d_state]))

    def learn(self, state, action, reward, next_state, done):
        d_state = self.discretize_state(state)
        d_next_state = self.discretize_state(next_state)
        best_next_q = np.max(self.q_table[d_next_state])
        target = reward + self.gamma * best_next_q * (not done)
        self.q_table[d_state + (action,)] += self.alpha * (
            target - self.q_table[d_state + (action,)]
        )

    def save(self, path):
        np.save(path, self.q_table)

    def load(self, path):
        self.q_table = np.load(path)
