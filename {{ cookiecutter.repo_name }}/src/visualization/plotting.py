import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def plot_histogram(data, bins=10, title="Histogram", xlabel="Value", ylabel="Frequency"):
    """
    Draws a histogram for the given data.

    Parameters:
    data (list or array-like): The input data to be plotted as a histogram.
    bins (int or sequence): Number of histogram bins to be used (default is 10).
    title (str): Title of the histogram.
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.

    # Example usage:
        data = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
        draw_histogram(data, bins=5, title='Sample Histogram', xlabel='Data Values', ylabel='Count')
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_moments_histogram(arr: np.ndarray, figsize: tuple = (10, 8), bins: int = 20):
    """
    Plot histograms of mean, variance, skewness, and kurtosis of the input array.

    Parameters:
        arr (np.ndarray): Input array.
        figsize (tuple): Size of the figure (width, height).
        bins (int): Number of bins for the histograms.

    Returns:
        None
    """
    # Calculate statistics
    mean = arr.mean()
    variance = arr.var()
    skewness = stats.skew(arr)
    kurtosis = stats.kurtosis(arr)

    # Plot histograms
    fig, axs = plt.subplots(2, 2, figsize=figsize)

    axs[0, 0].hist(mean, bins=bins, color="blue", alpha=0.7)
    axs[0, 0].set_title("Mean")
    axs[0, 0].axvline(x=0, color="grey", linestyle="--", label="zero mean")
    axs[0, 0].legend()

    axs[0, 1].hist(variance, bins=bins, color="green", alpha=0.7)
    axs[0, 1].set_title("Variance")
    axs[0, 1].axvline(x=1, color="grey", linestyle="--", label="unit variance")
    axs[0, 1].legend()

    axs[1, 0].hist(skewness, bins=bins, color="orange", alpha=0.7)
    axs[1, 0].set_title("Skewness")
    axs[1, 0].axvline(x=-0.5, color="grey", linestyle="--", label="normal skewness")
    axs[1, 0].axvline(x=0.5, color="grey", linestyle="--")
    axs[1, 0].legend()

    axs[1, 1].hist(kurtosis, bins=bins, color="red", alpha=0.7)
    axs[1, 1].set_title("Kurtosis")
    axs[1, 1].axvline(x=3, color="grey", linestyle="--", label="normal kurtosis")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


# Example usage:
# Assuming arr is your array of data
# plot_statistics_histograms(arr, figsize=(12, 10), bins=30)
