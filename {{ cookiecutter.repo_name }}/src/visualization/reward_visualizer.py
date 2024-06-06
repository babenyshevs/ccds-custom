from collections import defaultdict
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np

from src.data.reward_generator import RewardGenerator
from src.data.utilites import bootstrap


class RewardVisualizer:
    """
    Class to visualize how rewards evolve over time using RewardGenerator instance.
    """

    def __init__(self, reward_generator: RewardGenerator, figsize=(10, 6)):
        """
        Initialize the RewardVisualizer.

        Args:
            reward_generator (RewardGenerator): An instance of RewardGenerator class.
            figsize (Tuple[int, int]): Figure size for the plot. Default is (10, 6).
        """
        self.reward_generator = reward_generator
        self.figsize = figsize
        self.arm_ids = reward_generator.arm_configs.keys()

    def prepare_data(self, num_trials: int) -> None:
        """
        Generate rewards data for each arm.

        Args:
            num_trials (int): The number of trials to generate rewards for.
        """
        self.arm_reward_dict = defaultdict(list)
        for arm_id in self.arm_ids:
            self.arm_reward_dict[arm_id] = self.reward_generator.pull_arm_n_times(
                arm_id, num_trials
            )

    def visualize_rewards(self) -> None:
        """
        Visualize how rewards evolve over time.
        """
        plt.figure(figsize=self.figsize)
        markers = cycle(["o", "s", "^", "x", "+"])
        for arm, rewards in self.arm_reward_dict.items():
            plt.plot(np.arange(len(rewards)), rewards, marker=next(markers), label=arm)
        plt.xlabel("Trials")
        plt.ylabel("Reward")
        plt.title("Reward Evolution Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_reward_histogram(self, bootstraped=False) -> None:
        """
        Plot histograms of rewards for each arm on the same graph.

        Args:
            bootstraped (bool): Whether to bootstrap the rewards or not.
        """
        plt.figure(figsize=self.figsize)
        for arm_id, rewards in self.arm_reward_dict.items():
            if bootstraped:
                rewards = bootstrap(data=rewards, repeats=1000, sample="log")
            plt.hist(rewards, bins=20, alpha=0.5, label=arm_id)

        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Rewards ({'Bootstrapped' if bootstraped else 'Original'})")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_cumulative_rewards(self, log_scale=False) -> None:
        """
        Plot cumulative values of rewards received.
        """
        plt.figure(figsize=self.figsize)
        for arm, rewards in self.arm_reward_dict.items():
            plt.plot(np.arange(len(rewards)), np.cumsum(rewards), label=arm)
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Rewards Received")
        plt.title("Cumulative Rewards Received Over Time")
        plt.legend()
        plt.grid(True)
        if log_scale:
            plt.yscale("log")  # Set y-axis to logarithmic scale
        plt.show()
