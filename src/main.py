"""
Flappy Bird Q-러닝 학습 스크립트
- Q-테이블 학습 후 총 보상 확인
"""

from enum import Enum
from agents.dqn_agent import DQNAgent
from agents.q_learning_agent import QLearningAgent
from evaluators.simple_evaluator import SimpleEvaluator
from flappy_bird_env import FlappyBirdEnv
from trainers.dqn_trainer import DQNTrainer
from trainers.q_learning_trainer import QLearningTrainer


class Model(Enum):
    Q_LEARNING = "q_learning"
    DEEP_Q_NETWORK = "deep_q_network"


def main():
    env = FlappyBirdEnv(render=False)
    algo = Model.Q_LEARNING
    print(">>> sample state from env reset:", type(env.reset()), env.reset())

    if algo == Model.Q_LEARNING:
        agent = QLearningAgent(env=env)
        trainer = QLearningTrainer(agent=agent, env=env)

    elif algo == Model.DEEP_Q_NETWORK:
        agent = DQNAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
        )

        trainer = DQNTrainer(agent, env, episodes=1000, evaluator=SimpleEvaluator())

    else:
        raise ValueError("Unknown algorithm")

    evaluator = SimpleEvaluator()

    # 4. 학습
    print(f"===== TRAINING ({algo}) =====")
    trainer.train()

    # 5. 평가
    print(f"===== EVALUATING ({algo}) =====")
    results = evaluator.evaluate(agent, env, episodes=5)
    print("Evaluation results:", results)


# ================================
# Main 실행
# ================================
if __name__ == "__main__":
    main()
