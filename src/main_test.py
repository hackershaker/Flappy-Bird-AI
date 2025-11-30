"""
Q-러닝 학습 완료 후, 에이전트 플레이 및 시각화
- pygame으로 화면 출력
"""

import numpy as np
from flappy_bird_env import FlappyBirdEnv
from q_learning_agent import QLearningAgent

"""

try:
    Q_table = np.load("q_table.npy")
except:
    print("Q table 불러오기 실패 : ")

# 화면 초기화
pygame.init()
screen = pygame.display.set_mode((288, 512))
pygame.display.set_caption("Flappy Bird - Q-러닝 에이전트")
clock = pygame.time.Clock()

# 색상 정의
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BIRD_COLOR = (255, 0, 0)
PIPE_COLOR = (0, 255, 0)

# 환경 초기화
env = FlappyBirdEnv()
state_raw = env.reset()
state = discretize_state(state_raw, bins=state_bins)
done = False
score = 0

while not done:
    # 이벤트 처리 (창 닫기)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Q-테이블 기반 행동 선택 (탐험 없이)
    action = np.argmax(Q_table[state])
    next_state_raw, reward, done = env.step(action)
    next_state = discretize_state(next_state_raw, bins=state_bins)
    state = next_state
    score += reward

    # 화면 그리기
    screen.fill(WHITE)
    # 새
    pygame.draw.circle(screen, BIRD_COLOR, (50, int(env.bird_y)), 10)
    # 파이프 (상단/하단)
    pygame.draw.rect(screen, PIPE_COLOR, pygame.Rect(env.pipe_x, 0, 50, env.pipe_y))
    pygame.draw.rect(
        screen,
        PIPE_COLOR,
        pygame.Rect(env.pipe_x, env.pipe_y + env.pipe_gap, 50, env.height),
    )
    # 점수 표시
    font = pygame.font.SysFont(None, 24)
    text = font.render(f"Score: {score}", True, BLACK)
    screen.blit(text, (10, 10))

    pygame.display.flip()
    clock.tick(30)  # FPS

pygame.quit()
print(f"게임 종료! 총 점수: {score}")

"""


class QLearningTestAgent(QLearningAgent):
    def __init__(self, env, q_table_path="q_table.npy", bins=(10, 10, 10, 10)):
        super().__init__(env, bins=bins)
        # 학습된 Q-table 불러오기
        self.q_table = np.load(q_table_path)
        # epsilon=0: 무작위 행동 없음
        self.epsilon = 0.0


# ================================
# 테스트 실행
# ================================
if __name__ == "__main__":
    # 1. 환경 생성 (render=True)
    env = FlappyBirdEnv(render=True)

    # 2. 에이전트 생성 및 Q-table 로드
    agent = QLearningTestAgent(env, q_table_path="best_q_table.npy")
    agent.epsilon = 0.0

    # 3. 초기 상태
    state = env.reset()
    done = False
    total_reward = 0

    # 4. 게임 루프
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

        # 렌더링
        env.render()

    print(f"Game Over! Total Reward: {total_reward}")
