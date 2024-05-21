from typing import Union

from src.general.io import from_pickle, to_pickle
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


def download_transformers_model(
    model_name: str, save_path: Union[str, dict], save_pickle: bool = True
):
    """
    Download and save (optionaly) a Hugging Face Transformers model, tokenizer, and configuration for sequence classification,
    and save them in the specified directory.

    Args:
        model_name (str): The name or path of the pretrained model.
        save_path (Union[str, dict]): Destination path for saving. If dict is provided with keys 'tokenizer', 'model', 'config',
                                       the tokenizer, model, and config will be saved separately; otherwise, all components will be saved at once.
        save_pickle (bool, optional): If True, saves the components using pickle format. Defaults to True.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    model_params = {"tokenizer": tokenizer, "model": model, "config": config}

    if isinstance(save_path, dict):
        if "tokenizer" not in save_path or "model" not in save_path or "config" not in save_path:
            raise ValueError(
                "If save_path is a dictionary, it must contain keys: 'tokenizer', 'model', and 'config'."
            )
        if save_pickle:
            to_pickle(tokenizer, save_path["tokenizer"])
            to_pickle(model, save_path["model"])
            to_pickle(config, save_path["config"])
        else:
            tokenizer.save_pretrained(save_path["tokenizer"])
            model.save_pretrained(save_path["model"])
            config.save_pretrained(save_path["config"])
        return model_params
    elif isinstance(save_path, str):
        if save_pickle:
            to_pickle(model_params, save_path)
        else:
            tokenizer.save_pretrained(save_path)
            model.save_pretrained(save_path)
            config.save_pretrained(save_path)
        return model_params
    else:
        return model_params


def load_transformers_model(load_path: Union[str, dict], read_pickle: bool = True) -> dict:
    """
    Load a saved Hugging Face Transformers model, tokenizer, and configuration for sequence classification.

    Args:
        load_path (Union[str, dict]): Path for loading. If dict is provided with keys 'tokenizer', 'model', 'config',
                                       the tokenizer, model, and config will be loaded separately; otherwise, all components will be loaded at once.
        read_pickle (bool, optional): If True, reads the components using pickle format. Defaults to True.

    Returns:
        dict: A dictionary containing the tokenizer, model, and configuration.
    """
    if isinstance(load_path, dict):
        if "tokenizer" not in load_path or "model" not in load_path or "config" not in load_path:
            raise ValueError(
                "If load_path is a dictionary, it must contain keys: 'tokenizer', 'model', and 'config'."
            )
        tokenizer = (
            from_pickle(load_path["tokenizer"])
            if read_pickle
            else AutoTokenizer.from_pretrained(load_path["tokenizer"])
        )
        model = (
            from_pickle(load_path["model"])
            if read_pickle
            else AutoModelForSequenceClassification.from_pretrained(load_path["model"])
        )
        config = (
            from_pickle(load_path["config"])
            if read_pickle
            else AutoConfig.from_pretrained(load_path["config"])
        )
    else:
        model_params = from_pickle(load_path) if read_pickle else from_pickle(load_path)
        tokenizer = model_params["tokenizer"]
        model = model_params["model"]
        config = model_params["config"]
    return {"tokenizer": tokenizer, "model": model, "config": config}
