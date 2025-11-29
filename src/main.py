"""
Flappy Bird Q-러닝 학습 스크립트
- Q-테이블 학습 후 총 보상 확인
"""

import numpy as np
from flappy_bird import FlappyBirdEnv
from q_learning_agent import discretize_state

# 환경 및 파라미터 설정
env = FlappyBirdEnv()
state_bins = (10, 10, 10, 10)
action_space = [0, 1]

# Q-테이블 초기화
Q_table = np.zeros(state_bins + (len(action_space),))

# 학습 파라미터
alpha = 0.1       # 학습률
gamma = 0.99      # 할인율
epsilon = 0.1     # 탐험 확률
episodes = 5000   # 학습 반복 횟수

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
        Q_table[state + (action,)] += alpha * (reward + gamma * best_next - Q_table[state + (action,)])

        state = next_state
        total_reward += reward

    if episode % 500 == 0:
        print(f"Episode {episode}, Reward: {total_reward}")

# 학습 완료 후 테스트
print("학습 완료! 최종 Q-테이블 저장 가능")

# Q-테이블 저장
np.save("q_table.npy", Q_table)
