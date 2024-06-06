# Importing necessary libraries
import dash
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import dcc, html
from dash.dependencies import Input, Output

from src.data.utilites import bootstrap, get_marks

# Initializing the Dash app
app = dash.Dash(__name__)

# Defining the layout of the app
app.layout = html.Div(
    [
        dcc.Graph(id="graph"),
        html.P("Sample size"),
        dcc.Slider(id="sample", min=50, max=1500, value=300),
        html.P("Effect size"),
        dcc.Slider(id="bias", min=0, max=1, value=0.1, marks=get_marks(0, 1, 0.1, 1)),
    ]
)


# Callback function to update the graph based on user input
@app.callback(Output("graph", "figure"), [Input("sample", "value"), Input("bias", "value")])
def display_color(sample, bias):
    fig = go.Figure()
    fig.data = []

    for label, bs in zip(["A", "B", "C"], [0, bias, -bias]):
        samp = bootstrap(DATA, VARIABLE, bias=bs, repeats=1000, sample=sample)
        sample_mean = np.round(np.mean(samp), 2)
        fig.add_trace(go.Histogram(x=samp, name=f"bot {label}"))
        fig.add_vline(x=sample_mean, line_width=1, annotation_text=f"Mean: {sample_mean}")

    fig.update_layout(barmode="overlay")
    fig.update_layout(title=f"A, B, C groups - distribution of target metric ({VARIABLE})")
    fig.update_traces(opacity=0.75)
    fig.show()
    return fig


# Running the app if the script is executed directly
if __name__ == "__main__":

    DATA = pd.read_csv(r"data\processed\labstudio.csv")
    VARIABLE = "acceptable"  # fabricating_info

    app.run_server(debug=False)
