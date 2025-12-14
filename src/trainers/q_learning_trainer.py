from evaluators.simple_evaluator import SimpleEvaluator


class QLearningTrainer:
    def __init__(
        self,
        agent,
        env,
        episodes: int = 1000,
        eval_interval: int = 50,
        eval_episodes: int = 20,
        save_path: str = "best_q_table.npy",
    ):
        self.agent = agent
        self.env = env
        self.episodes = episodes
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        self.save_path = save_path

        self.evaluator = SimpleEvaluator()
        self.best_eval_avg = -float("inf")

    def train(self):
        for ep in range(self.episodes):
            state = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self.agent.act(state)

                next_state, reward, done = self.env.step(action)

                self.agent.learn(state, action, reward, next_state, done)

                state = next_state
                total_reward += reward

            print(
                f"[Train] EP {ep} | Reward: {total_reward:.2f} | "
                f"Epsilon: {getattr(self.agent, 'epsilon', None)}"
            )

            if (ep + 1) % self.eval_interval == 0:
                avg_reward, results = self.evaluator.evaluate(
                    self.agent,
                    self.env,
                    episodes=self.eval_episodes,
                    use_eval_act=True,
                )

                print(
                    f"[Eval] EP {ep} | "
                    f"Avg({self.eval_episodes} ep): {avg_reward:.3f} "
                    f"| Sample: {results}"
                )

                if avg_reward > self.best_eval_avg:
                    self.best_eval_avg = avg_reward
                    self.agent.save(self.save_path)
                    print(
                        f"🔥 New Best Q-table Saved! "
                        f"(Eval Avg: {avg_reward:.3f}) -> {self.save_path}"
                    )

        print(
            f"Training finished. "
            f"Best eval avg reward: {self.best_eval_avg:.3f} "
            f"(saved at {self.save_path})"
        )
