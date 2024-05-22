from typing import List

import spacy
from gensim.utils import simple_preprocess


def lemmatize(
    docs: List[str], allowed_postags: List[str] = ["NOUN", "ADJ", "VERB", "ADV"]
) -> List[str]:
    """
    Performs lemmization of input documents.

    Args:
        docs (List[str]): List of strings with input documents.
        allowed_postags (List[str], optional): List of accepted Part of Speech (POS) types. Defaults to ["NOUN", "ADJ", "VERB", "ADV"].

    Returns:
        List[str]: List of strings with lemmatized input.
    """
    # Load English language model for spaCy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    lemmatized_docs = []
    # Iterate through each document
    for doc in docs:
        doc = nlp(doc)
        tokens = []
        # Iterate through each token in the document
        for token in doc:
            # Check if the token's part of speech is in the allowed list
            if token.pos_ in allowed_postags:
                tokens.append(token.lemma_)
        # Join the lemmatized tokens to form the lemmatized document
        lemmatized_docs.append(" ".join(tokens))
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
        tokens = simple_preprocess(doc, deacc=True)
        tokenized_docs.append(tokens)
    return tokenized_docs
