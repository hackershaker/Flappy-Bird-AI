class SimpleEvaluator:
    def evaluate(self, agent, env, episodes=10, use_eval_act=True):
        """
        episodes 번 에피소드를 돌려 각 total_reward를 기록하고,
        (1) 에피소드별 리워드 리스트와
        (2) 평균 리워드
        를 함께 반환.
        """
        results = []
        for _ in range(episodes):
            state = env.reset()
            done = False
            total = 0

            while not done:
                if use_eval_act and hasattr(agent, "act_eval"):
                    action = agent.act_eval(state)
                else:
                    action = agent.act(state)

                state, reward, done = env.step(action)
                total += reward

            results.append(total)

        S = sum(results)
        avg = S / len(results) if results else 0.0

        return avg, results
