import torch


class DQNTrainer:
    def __init__(
        self,
        agent,
        env,
        evaluator,
        episodes=600,
        eval_interval=20,
        eval_episodes=10,
        device="cpu",
    ):
        self.agent = agent
        self.env = env
        self.evaluator = evaluator
        self.episodes = episodes
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.best_eval_avg = -float("inf")
        self.device = device

    def train(self):
        for ep in range(1, self.episodes + 1):
            state = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                # 행동 선택
                action = self.agent.act(state)

                # 환경 진행
                next_state, reward, done = self.env.step(action)

                # memorize
                self.agent.memorize(state, action, reward, next_state, done)

                # learn
                self.agent.learn()

                # update state
                state = next_state

                # 보상 누적
                total_reward += reward

            self.agent.decaying_epsilon()

            if (ep + 1) % self.eval_interval == 0:
                avg_reward, results = self.evaluator.evaluate(
                    self.agent,
                    self.env,
                    episodes=self.eval_episodes,
                    use_eval_act=True,
                )

                print(
                    f"[EVAL] EP {ep} | "
                    f"AvgReward({self.eval_episodes} ep): {avg_reward:.3f} "
                    f"| Sample: {results}"
                )

                # 최고 성능 모델 저장
                if avg_reward > self.best_eval_avg:
                    self.best_eval_avg = avg_reward
                    torch.save(
                        {
                            "policy_net": self.agent.policy_net.state_dict(),
                            "target_net": self.agent.target_net.state_dict(),
                            "epsilon": self.agent.epsilon,
                        },
                        "best_dqn_model.pth",
                    )
                    print(f"🔥 New Best Model Saved! (Eval Avg: {avg_reward:.3f})")
