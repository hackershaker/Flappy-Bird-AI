"""
Flappy Bird 환경 정의 (강화학습용)
- 상태: [새 y 위치, 속도, 파이프 x, 파이프 y]
- 행동: 0 = 아무것도 안함, 1 = 점프
- 보상: 살아있으면 +1, 충돌 시 -100
"""

import random
import numpy as np

class FlappyBirdEnv:
    def __init__(self, width=288, height=512):
        """
        Flappy Bird 환경 초기화
        :param width: 게임 화면 너비
        :param height: 게임 화면 높이
        """
        self.width = width
        self.height = height
        self.gravity = 1
        self.jump_velocity = -5
        self.pipe_gap = 100
        self.reset()

    def reset(self):
        """게임 초기화 및 상태 반환"""
        self.bird_y = self.height // 2
        self.bird_vel = 0 # 새의 수직 속도(1 스텝당 픽셀)
        self.pipe_x = self.width # 파이프의 가로 위치, 화면 오른쪽 끝에서 시작
        self.pipe_y = random.randint(50, self.height - 150) # 파이프 높이(랜덤하게 배치)
        self.done = False
        return self.get_state()

    def step(self, action):
        """
        한 스텝 환경 업데이트
        :param action: 0 = 아무것도 안함, 1 = 점프
        :return: (state, reward, done)
        """
        if action == 1:
            self.bird_vel = self.jump_velocity # 점프하면 수직속도 음수(위로 올라감)

        # 물리 적용
        self.bird_vel += self.gravity # 중력이 작용하여 아래로 떨어짐
        self.bird_y += self.bird_vel # 수직 속도에 따라 새의 좌표가 이동
        self.pipe_x -= 3 # 매 스텝마다 3픽셀씩 왼쪽으로 이동

        # 충돌 체크
        if self.bird_y <= 0 or self.bird_y >= self.height:
            self.done = True
        if self.pipe_x < 50 < self.pipe_x + 50 and not (self.pipe_y < self.bird_y < self.pipe_y + self.pipe_gap): # 새의 x좌표 50을 기준으로 충돌체크
            self.done = True

        # 보상 계산
        reward = 1 if not self.done else -100

        # 화면 왼쪽 밖으로 나가면 파이프 초기화 -> 다시 오른쪽으로 이동
        if self.pipe_x < -50:
            self.pipe_x = self.width
            self.pipe_y = random.randint(50, self.height - 150)

        return self.get_state(), reward, self.done

    def get_state(self):
        """현재 상태 반환"""
        return np.array([self.bird_y, self.bird_vel, self.pipe_x, self.pipe_y])
