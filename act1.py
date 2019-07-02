import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


class Bandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0., modify_epsilon=1000, temperature=1):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline

        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial
        self.modify_epsilon = modify_epsilon
        self.decay_temperature = temperature
        self.temperature = temperature

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.decay_temperature = self.temperature

    # get an action for this bandit
    def act(self):
        epsilon = self.epsilon

        if self.modify_epsilon is not None:
            epsilon = self.epsilon - (self.epsilon * (self.time % self.modify_epsilon) / self.modify_epsilon)
            if epsilon < 0:
                epsilon = 0

        if np.random.rand() < epsilon:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                             self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice([action for action, q in enumerate(UCB_estimation) if q == q_best])

        if self.gradient:
            exp_est = np.exp(self.q_estimation / self.decay_temperature)
            self.action_prob = exp_est / np.sum(exp_est)

            # decay the temp
            self.decay_temperature = self.decay_temperature * 0.99

            if self.decay_temperature < 0.1:
                self.decay_temperature = 0.1

            return np.random.choice(self.indices, p=self.action_prob)

        q_best = np.max(self.q_estimation)
        return np.random.choice([action for action, q in enumerate(self.q_estimation) if q == q_best])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        # print(self.temperature)
        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward


def simulate(runs, time, bandits):
    best_action_counts = np.zeros((len(bandits), runs, time))
    rewards = np.zeros(best_action_counts.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards


def bandit_definition():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig('images/bandit_figure.png')
    plt.show()


def activity_1_1(runs=2000, time=1000, initial=0.0):
    epsilons = [0.1, 0.15, 0.2]
    bandits = [Bandit(epsilon=eps, sample_averages=True, modify_epsilon=runs, initial=initial) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    plt.savefig('images/actividad_1_1.png')
    plt.show()


def activity_1_2(runs=2000, time=1000, initial=0.0):
    bandits = []
    bandits.append(Bandit(gradient=True, gradient_baseline=True, temperature=30.0, initial=initial))
    bandits.append(Bandit(gradient=True, gradient_baseline=True, temperature=50.0, initial=initial))
    bandits.append(Bandit(gradient=True, gradient_baseline=True, temperature=100.0, initial=initial))
    bandits.append(Bandit(gradient=True, gradient_baseline=True, temperature=1000.0, initial=initial))

    best_action_counts, _ = simulate(runs, time, bandits)
    labels = ['t = 30', 't = 50', 't = 100', 't = 1000']

    for i in range(len(bandits)):
        plt.plot(best_action_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()

    plt.savefig('images/actividad_1_2.png')
    plt.show()


def activity_1_3():
    activity_1_1(initial=0.3)
    activity_1_1(initial=0.5)
    activity_1_2(initial=0.3)
    activity_1_2(initial=0.5)


bandit_definition()
activity_1_1()
activity_1_2()
activity_1_3()
