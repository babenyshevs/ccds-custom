import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.data.reward_generator import RewardGenerator
from src.data.utilites import bootstrap
from src.general.utils import take_screenshot
from src.models.mab import MultiArmedBandit


class DynamicPlotApp:
    """
    A class to create a Streamlit app that dynamically plots rewards obtained by pulling an arm.

    Attributes:
    -----------
    rg : RewardGenerator
        An instance of RewardGenerator to generate rewards.
    bandit : MultiArmedBandit
        An instance of MultiArmedBandit to run the MAB algorithm.
    reward_var : str
        The variable name for rewards.
    attempt_var : str
        The variable name for attempts.
    algos : list[str]
        The list of algorithms being compared.
    arm_ids : list[str]
        The list of arm IDs.
    rewards : dict
        The rewards data.
    ab_data : dict
        The A/B test data.
    bandit_data : dict
        The Multi-armed bandit data.
    """

    def __init__(self, reward_config: dict, mab_config: dict) -> None:
        """
        Initialize the DynamicPlotApp with reward and MAB configurations.

        Parameters:
        -----------
        reward_config : dict
            Configuration for the reward generator.
        mab_config : dict
            Configuration for the multi-armed bandit algorithm.
        """
        self.reward_config = reward_config
        self.mab_config = mab_config
        self.rg = RewardGenerator(reward_config)
        self.bandit = MultiArmedBandit(self.rg, mab_config)
        self.reward_var = "Solution rate"
        self.attempt_var = "Trials"
        self.algos = ["A-B test", "Multi-armed bandit"]
        self.arm_ids = list(self.rg.arm_configs.keys())

        self.rewards = {}
        self.ab_data = {}
        self.bandit_data = {}

    def run(self) -> None:
        """
        Run the Streamlit app to visualize the rewards.
        """
        st.set_page_config(layout="wide")
        # st.title(f"Comparison of A/B test and MAB on {self.reward_var} data")

        # Sidebar for page selection
        page = st.sidebar.selectbox("Choose a page", ["Configuration", "Visualization"])

        if page == "Configuration":
            self.config_page()
        elif page == "Visualization":
            self.visualization_page()

    def config_page(self) -> None:
        """
        Display the configuration page to adjust the reward configuration.
        """
        st.header("Reward Configuration")

        # Initialize session state if not already done
        if "reward_config" not in st.session_state:
            st.session_state.reward_config = self.reward_config

        for arm_id, config in st.session_state.reward_config.items():
            with st.expander(f"Configuration for {arm_id}"):
                dist = st.selectbox(
                    "Distribution",
                    ["gauss", "uniform"],
                    index=["gauss", "uniform"].index(config["distribution"]),
                    key=f"dist_{arm_id}",
                )
                if dist == "gauss":
                    mean = st.number_input("Mean", value=config["params"][0], key=f"mean_{arm_id}")
                    std = st.number_input(
                        "Standard Deviation", value=config["params"][1], key=f"std_{arm_id}"
                    )
                    st.session_state.reward_config[arm_id] = {
                        "distribution": dist,
                        "params": [mean, std],
                    }
                elif dist == "uniform":
                    lower = st.number_input(
                        "Lower Bound", value=config["params"][0], key=f"lower_{arm_id}"
                    )
                    upper = st.number_input(
                        "Upper Bound", value=config["params"][1], key=f"upper_{arm_id}"
                    )
                    st.session_state.reward_config[arm_id] = {
                        "distribution": dist,
                        "params": [lower, upper],
                    }

        if st.button("Update Configuration"):
            self.rg = RewardGenerator(st.session_state.reward_config)
            st.success("Reward configuration updated!")

    def visualization_page(self) -> None:
        """
        Display the visualization page to show the rewards comparison.
        """
        # Settings
        st.sidebar.title("Controls")
        self.sleep_time = st.sidebar.radio(
            "Graph delay (seconds):", (0.001, 0.01, 0.1, 1), horizontal=True
        )
        st.session_state["cumsum"] = st.sidebar.checkbox("Show cumulative")
        st.session_state["bootstrap"] = st.sidebar.checkbox("Use bootstrap")
        st.sidebar.markdown("""---""")
        self.data_length = st.sidebar.number_input(
            "Trials per group:", value=100, min_value=10, max_value=2000
        )

        if st.sidebar.button("Reset trials"):
            st.session_state["iteration"] = 1

        st.sidebar.markdown("""---""")
        fit_share = st.sidebar.number_input(
            "Explore rounds, %:", value=30, min_value=1, max_value=99
        )
        self.mab_obs = self.data_length * len(self.arm_ids)  # because we pull only 1 arm per time
        self.num_fit_rounds = int(self.mab_obs * (fit_share / 100))
        self.num_next_steps = self.mab_obs - self.num_fit_rounds

        st.sidebar.write(
            f"""
            Rounds: {self.mab_obs} \n 
            Explore rounds: {self.num_fit_rounds} \n
            Exploit rounds: {self.num_next_steps} \n
            """
        )

        # Main code
        st.sidebar.markdown("""---""")
        if st.sidebar.button("Start"):
            self.generate_plot()

    def generate_data(self) -> None:
        """
        Generate the data for both A/B testing and multi-armed bandit algorithms.
        """
        self.data = {
            algo: {"y": {}, "y_cum": {}, "trials_cum": {}, "y_total": 0} for algo in self.algos
        }

        # A/B data (self.algos[0])
        for arm_id in self.arm_ids:
            arm_data = self.rg.pull_arm_n_times(arm_id, self.data_length)
            self.data[self.algos[0]]["y"][arm_id] = arm_data
            self.data[self.algos[0]]["y_cum"][arm_id] = np.cumsum(arm_data)
            self.data[self.algos[0]]["trials_cum"][arm_id] = np.arange(0, self.data_length + 1, 1)

        # Bandit data (self.algos[1])
        self.bandit.fit(self.num_fit_rounds)
        self.bandit.run_n_rounds(self.num_next_steps)
        self.data[self.algos[1]]["y"] = self.bandit.arm_reward_log
        self.data[self.algos[1]]["y_cum"] = self.bandit.arm_reward_cum_log
        self.data[self.algos[1]]["trials_cum"] = self.bandit.arm_pull_cum_log

    def get_scatter(self, alg_name: str) -> go.Figure:
        """
        Create a scatter plot for a given algorithm.

        Parameters:
        -----------
        alg_name : str
            The name of the algorithm.

        Returns:
        --------
        go.Figure
            The scatter plot figure.
        """
        fig = go.Figure()
        fig.update_layout(
            xaxis_title=self.attempt_var,
            yaxis_title=self.reward_var,
            # title=f"{alg_name}: achieved {self.reward_var}",
        )
        fig.update_yaxes(type="log")

        for arm_id in self.arm_ids:
            fig.add_trace(go.Scatter(x=[], y=[], mode="lines", marker=dict(size=8), name=arm_id))
        return fig

    def get_histogram(self, alg_name: str) -> go.Figure:
        """
        Create a histogram for a given algorithm.

        Parameters:
        -----------
        alg_name : str
            The name of the algorithm.

        Returns:
        --------
        go.Figure
            The histogram figure.
        """
        fig = go.Figure()
        fig.update_layout(
            xaxis_title=self.reward_var,
            barmode="overlay",
            # title=f"{alg_name}: histogram of {self.reward_var}",
        )
        for arm_id in self.arm_ids:
            fig.add_trace(
                go.Histogram(x=[], nbinsx=10, name=arm_id, opacity=0.5, histnorm="percent")
            )
        return fig

    def get_bar(self, alg_name: str) -> go.Figure:
        """
        Create a bar chart for a given algorithm.

        Parameters:
        -----------
        alg_name : str
            The name of the algorithm.

        Returns:
        --------
        go.Figure
            The bar chart figure.
        """
        fig = go.Figure()
        # fig.update_layout(xaxis_title=self.reward_var, title=f"{alg_name}: total trials")
        fig.update_yaxes(range=[0, self.mab_obs])
        for arm_id in self.arm_ids:
            fig.add_trace(
                go.Bar(
                    x=[],
                    y=[],
                    name=arm_id,
                    opacity=0.5,
                )
            )
        return fig

    def generate_plot(self) -> None:
        """
        Generate and update the line plot with rewards obtained by pulling the arm.
        """
        self.generate_data()

        # Plotly graphs
        go_scatters, go_histogtrams, go_bars = {}, {}, {}

        for alg_name in self.algos:
            go_scatters[alg_name] = self.get_scatter(alg_name)
            go_histogtrams[alg_name] = self.get_histogram(alg_name)
            go_bars[alg_name] = self.get_bar(alg_name)

        # Streamlit plots
        left_col, mid_col, right_col = st.columns(spec=[0.4, 0.3, 0.3], gap="large")
        st_scatters, st_histogtrams, st_bars = {}, {}, {}

        with left_col:
            for alg_name in self.algos:
                st_scatters[alg_name] = st.empty()

        with mid_col:
            for alg_name in self.algos:
                st_histogtrams[alg_name] = st.empty()

        with right_col:
            for alg_name in self.algos:
                st_bars[alg_name] = st.empty()

        # Update the plots with delay
        for iter in range(st.session_state.get("iteration", 1), self.mab_obs):

            # Interim results to display
            total_y = {alg: 0 for alg in self.algos}
            total_trials = {alg: 0 for alg in self.algos}

            # Update plotly plots
            for trace, arm_id in enumerate(self.arm_ids):

                # Scatters
                for alg_name, go_scatter in go_scatters.items():
                    y = self.data[alg_name]["y"][arm_id][: iter + 1]
                    y_cum = self.data[alg_name]["y_cum"][arm_id][: iter + 1]
                    total_y[alg_name] += y_cum[-1]
                    go_scatter.data[trace].x = list(range(0, iter))
                    go_scatter.data[trace].y = y_cum if st.session_state["cumsum"] else y

                # Histograms
                for alg_name, go_hist in go_histogtrams.items():
                    y_array = np.array(self.data[alg_name]["y"][arm_id][: iter + 1])
                    y_nozero = y_array[y_array != 0]  # Zeros, when arm is not pulled
                    if st.session_state["bootstrap"]:
                        y_nozero = bootstrap(y_nozero, repeats=self.data_length)
                    go_hist.data[trace].x = y_nozero

                # Bars
                for alg_name, go_bar in go_bars.items():
                    try:
                        y = self.data[alg_name]["trials_cum"][arm_id][iter]
                        total_trials[alg_name] += y
                    except IndexError:  # Happens because trials are (should be) imbalanced
                        y = np.array(self.data[alg_name]["trials_cum"][arm_id][self.data_length])
                        total_trials[alg_name] = self.mab_obs
                    go_bar.data[trace].x = [arm_id]
                    go_bar.data[trace].y = [y]

            # Update Streamlit plots
            for alg_name, st_scatter in st_scatters.items():
                # go_scatters[alg_name].update_layout(
                #     title=f"{alg_name}: achieved {self.reward_var} ({int(total_y[alg_name])})"
                # )
                st_scatter.plotly_chart(
                    go_scatters[alg_name],
                    use_container_width=True,
                )

            for alg_name, st_histogram in st_histogtrams.items():
                st_histogram.plotly_chart(go_histogtrams[alg_name], use_container_width=True)

            for alg_name, st_bar in st_bars.items():
                # go_bars[alg_name].update_layout(
                #     title=f"{alg_name}: total trials ({int(total_trials[alg_name])})"
                # )
                st_bar.plotly_chart(go_bars[alg_name], use_container_width=True)

            # # take a screenshot:
            # if iter % 5 == 0:
            #     take_screenshot(
            #         f"reports/figures/5_sec/{iter :03d}.png",
            #         left_trim=300,
            #         right_trim=50,
            #         top_trim=200,
            #         bottom_trim=75,
            #     )

            time.sleep(self.sleep_time)

        st.session_state["iteration"] = 1


if __name__ == "__main__":
    reward_config = {
        "superstar": {
            "distribution": "gauss",
            "params": [0.7, 0.05],
        },  # higher performant, unstable (E[X] = 0.7)
        "old_navy": {
            "distribution": "gauss",
            "params": [0.65, 0.02],
        },  # lower performant, stable (E[X] = 0.65)
        "ugly_duck": {
            "distribution": "uniform",
            "params": [0.60, 0.75],
        },  # reasonable random walker, (E[X] = 0.675)
    }

    mab_config = {"method": "epsilon_greedy", "method_params": {"epsilon": 0.1}}

    app = DynamicPlotApp(reward_config, mab_config)
    app.run()
