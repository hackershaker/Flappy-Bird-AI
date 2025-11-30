"""
Flappy Bird Q-러닝 학습 스크립트
- Q-테이블 학습 후 총 보상 확인
"""

import copy
import numpy as np
from base_agent import BaseAgent
from flappy_bird_env import FlappyBirdEnv
from q_learning_agent import QLearningAgent


"""

# 환경 및 파라미터 설정
env = FlappyBirdEnv()
state_bins = (10, 10, 10, 10)
action_space = [0, 1]

# Q-테이블 초기화
Q_table = np.zeros(state_bins + (len(action_space),))

# 학습 파라미터
alpha = 0.1  # 학습률
gamma = 0.99  # 할인율
epsilon = 0.1  # 탐험 확률
episodes = 5000  # 학습 반복 횟수

# 학습 루프
for episode in range(episodes):
    state = discretize_state(env.reset(), bins=state_bins)
    done = False
    total_reward = 0

    while not done:
        # epsilon-greedy 행동 선택
        if np.random.rand() < epsilon:
            action = np.random.choice(action_space)
        else:
            action = np.argmax(Q_table[state])

        # 환경 한 스텝 진행
        next_state_raw, reward, done = env.step(action)
        next_state = discretize_state(next_state_raw, bins=state_bins)

        # Q-테이블 업데이트
        best_next = np.max(Q_table[next_state])
        Q_table[state + (action,)] += alpha * (
            reward + gamma * best_next - Q_table[state + (action,)]
        )

        state = next_state
        total_reward += reward

    if episode % 500 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")

# 학습 완료 후 테스트
print("학습 완료! 최종 Q-테이블 저장 가능")

# Q-테이블 저장
np.save("q_table.npy", Q_table)


"""


# ================================
# Training Loop
# ================================
class FlappyBirdTrainer:
    def __init__(self, agent: BaseAgent, episodes=1000):
        self.env = FlappyBirdEnv(render=False)
        self.agent = agent  # 모델 에이전트( ex: Q-Learning, DQN, PPO... )
        self.episodes = episodes  # 총 학습 반복 수
        self.best_reward = -np.inf  # 가장 좋은 총 보상
        self.best_q_table = None  # best 모델 저장용
        self.best_steps = 0

    def train(self):
        for ep in range(self.episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            current_steps = 0

            while not done:
                action = self.agent.act(state)
                next_state, reward, done = self.env.step(action)
                self.agent.learn(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward
                current_steps += 1

            if (ep + 1) % 100 == 0:
                print(
                    f"Episode {ep+1}/{self.episodes}, Total Reward: {total_reward}, Total step: {current_steps}"
                )

            self._update_best_model(current_steps, ep)

        # self.agent.save("q_table.npy")
        print("Training completed!")

    def _update_best_model(self, steps, episode):
        """best Q-table을 reward 기준으로 갱신"""
        if episode < 20:  # warm-up 20 episodes
            return

        if steps > self.best_steps:
            print(f"✨ BEST MODEL UPDATED — steps survived: {steps}")
            self.best_steps = steps
            self.best_q_table = copy.deepcopy(self.agent.q_table)
            np.save("./best_q_table.npy", self.best_q_table)


# ================================
# Main 실행
# ================================
if __name__ == "__main__":
    agent: BaseAgent = QLearningAgent(env=None)
    trainer = FlappyBirdTrainer(agent=agent, episodes=5000)
    trainer.train()
