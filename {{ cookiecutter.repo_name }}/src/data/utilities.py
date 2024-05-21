import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import pandas as pd
import sweetviz as sv


def eda_report(
    data: pd.DataFrame,
    filename: str,
    skip: list = None,
    force_cat: list = None,
    force_num: list = None,
    target: str = None,
) -> sv.DataframeReport:
    """
    Generate an Exploratory Data Analysis (EDA) report using Sweetviz.

    Args:
        data (DataFrame): The input DataFrame for analysis.
        filename (str): The filename (including path) to save the HTML report.
        skip (list, optional): List of column names to skip during analysis. Default is None.
        force_cat (list, optional): List of column names to force treat as categorical. Default is None.
        force_num (list, optional): List of column names to force treat as numerical. Default is None.
        target (str, optional): The target column for which analysis will be performed. Default is None.

    Returns:
        sv.DataframeReport: The Sweetviz DataframeReport object containing the analysis.

    """
    feat_cfg = sv.FeatureConfig(skip=skip, force_cat=force_cat, force_num=force_num)
    report = sv.analyze(data, target_feat=target, feat_cfg=feat_cfg)
    report.show_html(filepath=filename, open_browser=False)
    return report


def get_lime_explanation(dataset, pipeline, instance_index, num_features=5):
    """
    Generate a LIME explanation picture for a given instance in the dataset for regression tasks.

    Parameters:
        dataset (numpy array or pandas DataFrame): The dataset used for training the pipeline.
        pipeline (scikit-learn Pipeline): The trained scikit-learn pipeline.
        instance_index (int): Index of the instance in the dataset for which explanation is needed.
        num_features (int, optional): Number of features to include in the explanation. Default is 5.

    Returns:
        matplotlib figure: Lime explanation picture.
    """

    transformed = pipeline["preprocessor"].transform(dataset)

    numerical_features = pipeline.named_steps["preprocessor"].transformers_[0][2]
    categorical_features = pipeline.named_steps["preprocessor"].transformers_[1][2]
    feature_names = numerical_features + list(
        pipeline.named_steps["preprocessor"]
        .named_transformers_["cat"]
        .get_feature_names_out(categorical_features)
    )

    explainer = lime.lime_tabular.LimeTabularExplainer(
        transformed, feature_names=feature_names, mode="regression", discretize_continuous=False
    )

    instance = transformed[instance_index]
    explanation = explainer.explain_instance(
        instance, pipeline["regressor"].predict, num_features=num_features
    )

    fig = explanation.as_pyplot_figure()
    plt.close()

    return fig
