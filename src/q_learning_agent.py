"""
Q-러닝 에이전트 구현
- 상태를 이산화(discretize) 후 Q-테이블 학습
- 연속 상태 -> Q-테이블 인덱스로 변환
"""

import numpy as np

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
