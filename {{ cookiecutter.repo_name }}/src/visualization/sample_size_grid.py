from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
from scipy.stats import norm
from statsmodels.stats.power import TTestIndPower

from src.data.utilites import bootstrap


class SampleSizeGrid:
    def __init__(
        self,
        x_name: str = "minimum detectable effect size",
        y_name: str = "observations needed (precise)",
        variables: list = None,
        data: Any = None,
        power: float = 0.8,
        alpha: float = 0.05,
    ):
        self.x_name = x_name
        self.y_name = y_name
        self.variables = variables
        self.data = data
        self.power = power
        self.alpha = alpha

    def calculate_observation_requirements(self, variable: str) -> pd.DataFrame:
        """
        Calculate observation requirements based on bootstrap analysis for a single variable.

        Args:
            VARIABLE (str): The variable of interest.

        Returns:
            pd.DataFrame: A DataFrame containing the observation requirements for the specified variable.
        """
        A = bootstrap(data=self.data, variable=variable, bias=0, repeats=1000, sample=100)
        m = np.mean(A)
        effects = {self.x_name: [], self.y_name: []}

        for effect_size in np.arange(start=0.01, stop=0.15, step=0.01):
            proportional_effect = sms.proportion_effectsize(m + effect_size, m)
            power_analysis = TTestIndPower()
            required_n = power_analysis.solve_power(
                effect_size=proportional_effect, power=self.power, alpha=self.alpha, nobs1=None
            )
            required_n = int(required_n)
            effects[self.x_name].append(effect_size)
            effects[self.y_name].append(required_n)

        df = pd.DataFrame(effects)
        return df

    def plot_observation_requirements(self) -> None:
        """
        Plot observation requirements for all variables.

        """
        plt.figure(figsize=(8, 6))

        for variable in self.variables:
            df = self.calculate_observation_requirements(variable)
            x = df[self.x_name]
            y = df[self.y_name]
            plt.plot(x, y, marker="o", linestyle="-", label=variable)

        plt.title("Observations Needed vs. Minimum Detectable Effect Size")
        plt.xlabel("Minimum Detectable Effect Size")
        plt.ylabel("Observations Needed (Precise)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_distributions(self) -> None:
        """
        Plot distributions obtained from calculate_observation_requirements for all variables.
        """
        plt.figure(figsize=(10, 6))

        for variable in self.variables:
            sample = bootstrap(data=self.data, variable=variable, bias=0, repeats=1000, sample=500)
            mu, std = norm.fit(sample)
            plt.hist(
                sample,
                bins=30,
                density=True,
                alpha=0.5,
                label=f"{variable} (std={np.round(std, 4)})",
            )

        plt.title("Distribution of metric mean")
        plt.xlabel("Mean")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage:
# analyzer = SampleSizeGrid(variables=["acceptable", "fabricating_info"], data=labstudio_data)
# analyzer.plot_observation_requirements()
