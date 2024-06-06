import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.stats import expon, norm, poisson, uniform


class DistributionVisualizer:
    def __init__(self):
        self.distribution_type = st.sidebar.selectbox(
            "Select Distribution", ("Normal", "Poisson", "Exponential", "Uniform")
        )

    def visualize_distribution(self):
        st.title("Distribution Visualizer")

        if self.distribution_type == "Normal":
            self.visualize_normal_distribution()
        elif self.distribution_type == "Poisson":
            self.visualize_poisson_distribution()
        elif self.distribution_type == "Exponential":
            self.visualize_exponential_distribution()
        elif self.distribution_type == "Uniform":
            self.visualize_uniform_distribution()

    def visualize_normal_distribution(self):
        st.header("Normal Distribution")
        mean = st.sidebar.number_input("Mean", value=0.0)
        std_dev = st.sidebar.number_input("Standard Deviation", value=1.0)
        x = np.linspace(-5, 5, 1000)
        y = norm.pdf(x, mean, std_dev)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("Normal Distribution")
        ax.set_xlabel("X")
        ax.set_ylabel("Probability Density")
        st.pyplot(fig)

    def visualize_poisson_distribution(self):
        st.header("Poisson Distribution")
        rate = st.sidebar.number_input("Rate (λ)", value=1.0)
        x = np.arange(0, 10)
        y = poisson.pmf(x, rate)
        fig, ax = plt.subplots()
        ax.bar(x, y)
        ax.set_title("Poisson Distribution")
        ax.set_xlabel("X")
        ax.set_ylabel("Probability Mass")
        st.pyplot(fig)

    def visualize_exponential_distribution(self):
        st.header("Exponential Distribution")
        rate = st.sidebar.number_input("Rate (λ)", value=1.0)
        x = np.linspace(0, 5, 1000)
        y = expon.pdf(x, scale=1 / rate)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("Exponential Distribution")
        ax.set_xlabel("X")
        ax.set_ylabel("Probability Density")
        st.pyplot(fig)

    def visualize_uniform_distribution(self):
        st.header("Uniform Distribution")
        start = st.sidebar.number_input("Start", value=0.0)
        end = st.sidebar.number_input("End", value=1.0)
        x = np.linspace(start, end, 1000)
        y = uniform.pdf(x, start, end - start)
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_title("Uniform Distribution")
        ax.set_xlabel("X")
        ax.set_ylabel("Probability Density")
        st.pyplot(fig)


if __name__ == "__main__":
    app = DistributionVisualizer()
    app.visualize_distribution()
