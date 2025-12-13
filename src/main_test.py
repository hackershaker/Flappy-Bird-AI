"""
에이전트 종류 선택 → Play 클래스 실행
"""

from players.dqn_play import DQNPlay
from players.q_learning_play import QLearningPlay


MODE = "Q"  # "Q" 또는 "DQN"


def main():
    if MODE == "Q":
        player = QLearningPlay(q_table_path="best_q_table.npy")
    elif MODE == "DQN":
        player = DQNPlay(model_path="best_dqn_model.pth", device="cpu")
    else:
        raise ValueError("MODE must be 'Q' or 'DQN'.")

    player.run()


if __name__ == "__main__":
    main()
