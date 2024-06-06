from typing import List, Optional, Union

import pandas as pd
import spacy
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess


def lemmatize(
    docs: List[str],
    model_name: str = "en_core_web_sm",
    disallowed_postags: List[str] = [
        "ART",
        "AUX",
        "ADP",
        "CCONJ",
        "INTJ",
        "NUM",
        "PDAT",
        "PDS",
        "PRON",
        "PUNCT",
        "SCONJ",
        "SYM",
        "X",
    ],
) -> List[str]:
    """
    Performs lemmization of input documents using spacy (see README if not installed)

    Args:
        docs (List[str]): List of strings with input documents.
        model_name (str, optional): Name of the spaCy language model to load. Defaults to "en_core_web_sm".
        disallowed_postags (List[str], optional): List of accepted Part of Speech (POS) types.
        full list is available at https://universaldependencies.org/u/pos/ and
        https://www.sketchengine.eu/german-stts-part-of-speech-tagset/

    Returns:
        List[str]: List of strings with lemmatized input.
    """
    # Load the specified language model for spaCy
    nlp = spacy.load(model_name, disable=["parser", "ner"])
    lemmatized_docs = []
    # Iterate through each document
    for doc in docs:
        doc = nlp(doc)
        tokens = []
        # Iterate through each token in the document
        for token in doc:
            # Check if the token's part of speech is in the allowed list
            if token.pos_ not in disallowed_postags:
                tokens.append(token.lemma_)
        # Join the lemmatized tokens to form the lemmatized document
        lemmatized_doc = " ".join(tokens)

        # If result is not an empty list, append it
        if lemmatized_doc:
            lemmatized_docs.append(lemmatized_doc)
    return lemmatized_docs


def tokenize(docs: List[str]) -> List[List[str]]:
    """
    Performs tokenization of input documents.

    Args:
        docs (List[str]): List of strings with input documents.

    Returns:
        List[List[str]]: List of lists of strings with tokenized input.
    """
    tokenized_docs = []
    # Iterate through each document
    for doc in docs:
        # Tokenize the document using Gensim's simple_preprocess
        tokens = simple_preprocess(doc, deacc=True, max_len=40)  # keep superlong german words
        tokenized_docs.append(tokens)
    return tokenized_docs


def generate_ngrams(sentences: list[list[str]], n: int) -> list[list[str]]:
    """
    Generate n-grams from a list of tokenized sentences.

    Args:
        sentences (list[list[str]]): List of tokenized sentences.
        n (int): The order of n-grams to generate (e.g., 2 for bigrams, 3 for trigrams).

    Returns:
        list[list[str]]: List of sentences with n-grams included.

    Example:
        >>> sentences = [["this", "is", "a", "sentence"], ["another", "sentence"]]
        >>> generate_ngrams(sentences, 3)
        [['this', 'is', 'a', 'sentence'], ['another', 'sentence']]
    """
    if n < 2:
        raise ValueError("n must be 2 or greater to form n-grams")

    current_sentences = sentences

    for _ in range(2, n + 1):
        phrases = Phrases(current_sentences, min_count=5, threshold=10)
        phraser = Phraser(phrases)
        current_sentences = [phraser[sentence] for sentence in current_sentences]

    return current_sentences


def filter_text_column(
    df: pd.DataFrame,
    column: str,
    filter_numbers: bool = True,
    starts_with: Optional[List[str]] = ["$", "/"],
) -> pd.DataFrame:
    """
    Remove rows from a DataFrame where the specified text column meets any of the following conditions:
    1. Contains only a number (if filter_numbers is True).
    2. Starts with any string in the starts_with list.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the text column to filter.
        filter_numbers (bool): Whether to filter rows containing only numbers. Default is True.
        starts_with (Optional[List[str]]): A list of strings. Rows starting with any of these strings will be filtered.
                                          Default is ['$', '/'].

    Returns:
        pd.DataFrame: The filtered DataFrame.

    Example:
        >>> data = {'text': ['123', '$money', '/path', 'valid text']}
        >>> df = pd.DataFrame(data)
        >>> filtered_df = filter_text_column(df, 'text')
        >>> print(filtered_df)
                text
        3  valid text

        >>> filtered_df = filter_text_column(df, 'text', filter_numbers=False, starts_with=['$', '#'])
        >>> print(filtered_df)
                text
        0        123
        3  valid text
    """

    def condition(text: str) -> bool:
        # Check if the text contains only a number
        if filter_numbers and text.isdigit():
            return True
        # Check if the text starts with any of the specified strings
        if any(text.startswith(prefix) for prefix in starts_with):
            return True
        return False

    # Apply the condition to the DataFrame and filter rows
    filtered_df = df[~df[column].apply(condition)].reset_index(drop=True)
    return filtered_df


def filter_rows_with_string(
    df: pd.DataFrame, target_string: str, column_spec: Union[str, List[str]]
) -> pd.DataFrame:
    """
    Filter out rows where the target string is present at the beginning of the string,
    followed by an underscore or space, in any columns specified by a prefix or a list of column names.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    target_string : str
        The string to search for in the columns.
    column_spec : Union[str, List[str]]
        Either a prefix to match the column names against or a list of exact column names.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the rows filtered out where the target string is found in the specified columns.

    Example
    -------
    >>> data = {
    >>>     "top_intent_A": ["abc", "def", "tks"],
    >>>     "top_intent_B": ["xyz", "tks", "ghi"],
    >>>     "other_column": ["123", "456", "789"]
    >>> }
    >>> df = pd.DataFrame(data)
    >>> filtered_df = filter_rows_with_string(df, "tks", "top_intent_")
    >>> print(filtered_df)
      top_intent_A top_intent_B other_column
    0          abc          xyz          123
    >>> filtered_df = filter_rows_with_string(df, "tks", ["top_intent_A", "other_column"])
    >>> print(filtered_df)
      top_intent_A top_intent_B other_column
    0          abc          xyz          123
    1          def          tks          456
    """
    # Build the regex pattern
    pattern = rf"(?:(?:^|[_\s]){target_string}(?:$|[_\s]))"

    if isinstance(column_spec, str):
        # Filter the columns based on the prefix
        columns_to_check = [col for col in df.columns if col.startswith(column_spec)]
    elif isinstance(column_spec, list):
        # Use the provided list of exact column names
        columns_to_check = [col for col in column_spec if col in df.columns]
    else:
        raise ValueError("column_spec must be either a string (prefix) or a list of column names")

    # Create a boolean mask for rows to keep
    mask = df[columns_to_check].astype(str).apply(lambda x: x.str.contains(pattern)).any(axis=1)

    # Filter the DataFrame using the mask
    filtered_df = df[mask]
    return filtered_df
