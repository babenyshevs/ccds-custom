import matplotlib.pyplot as plt

from src.models.mab import MultiArmedBandit


class BanditVisualizer:
    def __init__(
        self, bandit: MultiArmedBandit, num_fit_rounds: int, num_next_steps: int, figsize=(12, 8)
    ):
        self.bandit = bandit
        self.num_fit_rounds = num_fit_rounds
        self.num_next_steps = num_next_steps
        self.figsize = figsize

        # fit bandit and generate data
        self.bandit.fit(self.num_fit_rounds)
        self.bandit.run_n_rounds(self.num_next_steps)

    def plot_cumulative_values(self, log_scale=False):
        arms_cum_values = self.bandit.arm_pull_cum_log
        reward_cum_values = self.bandit.arm_reward_cum_log
        expectations = self.bandit.expectations_log

        fig, axs = plt.subplots(3, 1, figsize=self.figsize, sharex=True)

        for arm, values in arms_cum_values.items():
            axs[0].plot(values, label=arm)
        axs[0].axvline(self.num_fit_rounds, color="gray", linestyle="--", label="end of fitting")
        axs[0].set_ylabel("Cumulative Arm Value")
        axs[0].set_title("Evolution of Cumulative Arm Values")
        if log_scale:
            axs[0].set_yscale("log")
        axs[0].legend()

        for arm, values in reward_cum_values.items():
            axs[1].plot(values, label=arm)
        axs[1].axvline(self.num_fit_rounds, color="gray", linestyle="--", label="end of fitting")
        axs[1].set_ylabel("Cumulative Reward Value")
        axs[1].set_title("Evolution of Cumulative Reward Values")
        if log_scale:
            axs[1].set_yscale("log")
        axs[1].legend()

        for arm, values in expectations.items():
            axs[2].plot(values, label=arm)
        axs[2].set_xlabel("Round")
        axs[2].axvline(self.num_fit_rounds, color="gray", linestyle="--", label="end of fitting")
        axs[2].set_ylabel("Expected Reward")
        axs[2].set_title("Expected Reward Over Time")
        axs[2].legend()

    def plot_expected_histogram(self):
        plt.figure()
        expectations = self.bandit.expectations_log
        for arm, values in expectations.items():
            plt.hist(values, bins=100, alpha=0.5, label=arm)
        plt.title("Histogram of Expected Values")
        plt.xlabel("Expected Reward")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()


# # Example usage:
# # Assume you have already created a MultiArmedBandit instance called 'bandit'
# visualizer = BanditVisualizer(bandit, num_fit_rounds=1000, num_next_steps=100, figsize=(12, 8))
# visualizer.visualize()
