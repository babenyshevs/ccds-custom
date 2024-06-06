import logging
import pickle
import warnings
from typing import Any, Dict

import pyautogui
import urllib3
import yaml
from pandas.errors import SettingWithCopyWarning


def disable_warnings() -> None:
    """
    Disable various warnings.
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


def read_yaml(file_path: str) -> Dict[str, Any]:
    """
    Reads YAML data from a file and returns it as a dictionary.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: A dictionary containing the YAML data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error while parsing the YAML file.
    """
    with open(file_path, "r", encoding="utf-8") as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data


def to_pickle(file, filename):
    """ "
    Saves given python object as binary file (handy to avoid problems with types etc)
    file: file object (e.g. dataframe)
    filename: saving destination (path + filename withou extention), str
    return: True (deafault)
    """
    if ".pkl" not in filename:
        filename = f"{filename}.pkl"
    with open(filename, "wb") as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return True


def from_pickle(filename):
    """ "
    Reads and returns a binary file
    filename: loading destination (path + filename withou extention), str
    return: python object (whatever was saved: dict, dataframe, etc)
    """
    if ".pkl" not in filename:
        filename = f"{filename}.pkl"
    with open(filename, "rb") as handle:
        file = pickle.load(handle)
    return file


def get_logger():
    """
    gets default logger
    """
    loglevel = logging.INFO  # logging.DEBUG

    logger = logging.getLogger()
    logging.basicConfig(level=loglevel, force=True)
    logger.info(f"Log level set to {loglevel}")

    return logger


def take_screenshot(
    save_path: str,
    left_trim: int = 0,
    right_trim: int = 0,
    top_trim: int = 0,
    bottom_trim: int = 0,
):
    """
    Takes a single screenshot of a specified region on the screen.

    Args:
        save_path (str): The file path to save the screenshot.
        left_trim (int): Number of pixels to trim from the left side of the screenshot.
        right_trim (int): Number of pixels to trim from the right side of the screenshot.
        top_trim (int): Number of pixels to trim from the top of the screenshot.
        bottom_trim (int): Number of pixels to trim from the bottom of the screenshot.

    Example:
        take_screenshot((0, 0, 1920, 1080), 'screenshot.png', left_trim=100, right_trim=50, top_trim=0, bottom_trim=0)
    """
    width, height = pyautogui.size()
    screenshot = pyautogui.screenshot(
        region=(
            left_trim,
            top_trim,
            width - left_trim - right_trim,
            height - top_trim - bottom_trim,
        )
    )
    screenshot.save(save_path)
    print(f"Screenshot saved: {save_path}")
