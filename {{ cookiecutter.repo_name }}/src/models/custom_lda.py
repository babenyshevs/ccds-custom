from typing import Dict, List, Tuple, Union

import gensim
import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
import tqdm


class CustomLDA:
    """
    Class for fitting and visualizing Latent Dirichlet Allocation (LDA) models.

    Args:
        tokens (List[List[str]]): List of tokenized documents.
        seed (int): Random seed for reproducibility.

    Attributes:
        tokens (List[List[str]]): List of tokenized documents.
        seed (int): Random seed for reproducibility.
        id2word (gensim.corpora.Dictionary): Mapping from word IDs to words.
        corpus (Dict[str, List[Tuple[int, int]]]): Dictionary containing Document-Term Matrices.
        model (gensim.models.ldamodel.LdaModel): Fitted LDA model.

    Example:
        tokens = [["token1", "token2", ...], ["token1", "token3", ...], ...]
        lda = CustomLDA(tokens, seed=42)
        lda.fit()
        lda.visualize("train")
        coherence_score = lda.calculate_coherence_score(num_topics=5, alpha='auto', eta='auto')
    """

    def __init__(self, tokens: List[List[str]], seed: int):
        """
        Initialize CustomLDA with tokens and seed.

        Args:
            tokens (List[List[str]]): List of tokenized documents.
            seed (int): Random seed for reproducibility.

        Example:
            tokens = [["token1", "token2"], ["token3", "token4"]]
            lda = CustomLDA(tokens, seed=42)
        """
        self.tokens = tokens
        self.seed = seed
        self.id2word = None
        self.corpus = {"train": None}
        self.model = None
        self.best_num_topics = 30  # Default value to be determined later
        self.best_alpha = "symmetric"  # Default value to be determined later
        self.best_beta = None  # Default value to be determined later
        self.generate_id2word()
        self.corpus["train"] = self.generate_corpus(self.tokens)

    def generate_id2word(self) -> None:
        """
        Generate mapping from word IDs to words.

        Returns:
            None

        Example:
            lda = CustomLDA(tokens, seed=42)
            lda.generate_id2word()
        """
        self.id2word = gensim.corpora.Dictionary(self.tokens)

    def generate_corpus(self, tokens: List[List[str]]) -> List[List[Tuple[int, int]]]:
        """
        Generate Document-Term Matrix for the given tokens.

        Args:
            tokens (List[List[str]]): List of tokenized documents.

        Returns:
            List[List[Tuple[int, int]]]: Document-Term Matrix.

        Example:
            lda = CustomLDA(tokens, seed=42)
            dtm = lda.generate_corpus(tokens=[["token1", "token2"], ["token3", "token4"]])
        """
        return [self.id2word.doc2bow(doc) for doc in tokens]

    def add_corpus(self, tokens: List[List[str]], corpus_name: str) -> None:
        """
        Add a new corpus to the corpus dictionary.

        Args:
            tokens (List[List[str]]): List of tokenized documents.
            corpus_name (str): Name of the corpus to be added.

        Returns:
            None

        Example:
            lda = CustomLDA(tokens, seed=42)
            new_tokens = [["token5", "token6"], ["token7", "token8"]]
            lda.add_corpus(new_tokens, "new_corpus")
        """
        self.corpus[corpus_name] = self.generate_corpus(tokens)

    def fit(
        self,
        num_topics: int = None,
        alpha: Union[str, List[Union[float, str]]] = None,
        eta: Union[str, List[Union[float, str]]] = None,
    ) -> None:
        """
        Fit the LDA model.

        Args:
            num_topics (int, optional): Number of latent topics to extract. If not provided,
                                        the best_num_topics instance variable will be used.
            alpha (str or list, optional): Document-topic density. Default is None.
                                           If not provided, the best_alpha instance variable will be used.
            eta (str or list, optional): Topic-word density. Default is None.
                                         If not provided, the best_beta instance variable will be used.

        Returns:
            None

        Example:
            lda = CustomLDA(tokens, seed=42)
            lda.fit(num_topics=10, alpha='auto', eta='auto')
        """
        if num_topics is None:
            num_topics = self.best_num_topics
        if alpha is None:
            alpha = self.best_alpha
        if eta is None:
            eta = self.best_beta

        self.model = gensim.models.ldamodel.LdaModel(
            corpus=self.corpus["train"],
            id2word=self.id2word,
            num_topics=num_topics,
            alpha=alpha,
            eta=eta,
            random_state=self.seed,
            passes=100,
        )

    def visualize(self, corpus_name: str) -> pyLDAvis.PreparedData:
        """
        Visualize the LDA model for the specified corpus using pyLDAvis.

        Args:
            corpus_name (str): Name of the corpus to visualize.

        Returns:
            pyLDAvis.PreparedData: Visualization of the LDA model.

        Example:
            lda = CustomLDA(tokens, seed=42)
            lda.fit()
            lda_visual = lda.visualize("train")
        """
        pyLDAvis.enable_notebook()
        visualization = pyLDAvis.gensim_models.prepare(
            self.model, self.corpus[corpus_name], self.id2word, mds="mmds", R=21
        )

        return visualization

    def calculate_coherence_score(
        self,
        num_topics: int,
        alpha: Union[str, List[Union[float, str]]] = "auto",
        eta: Union[str, List[Union[float, str]]] = "auto",
    ) -> float:
        """
        Calculate coherence score for the LDA model.

        Args:
            num_topics (int): Number of latent topics to extract.
            alpha (str or list): Document-topic density. Default is 'auto'.
            eta (str or list): Topic-word density. Default is 'auto'.

        Returns:
            float: Coherence score.

        Example:
            lda = CustomLDA(tokens, seed=42)
            coherence_score = lda.calculate_coherence_score(num_topics=10, alpha='auto', eta='auto')
        """
        # Fit the LDA model using the fit method
        self.fit(num_topics=num_topics, alpha=alpha, eta=eta)

        # Create coherence model and calculate coherence score
        coherence_model_lda = gensim.models.CoherenceModel(
            model=self.model,
            texts=self.tokens,
            dictionary=self.id2word,
            coherence="c_v",
        )
        coherence_score = coherence_model_lda.get_coherence()
        return coherence_score

    def grid_search_hyperparameters(
        self,
        n_topics: Dict[str, int] = {"min": 10, "max": 100, "step": 10},
        alpha: Dict[str, float] = {"min": 0.01, "max": 1, "step": 0.3},
        beta: Dict[str, float] = {"min": 0.01, "max": 1, "step": 0.3},
    ) -> None:
        """
        Perform hyperparameters grid search.

        Args:
            n_topics (dict): Dictionary containing min, max, and step for number of topics.
            alpha (dict): Dictionary containing min, max, and step for alpha values.
            beta (dict): Dictionary containing min, max, and step for beta values.

        Returns:
            None

        Example:
            lda = CustomLDA(tokens, seed=42)
            lda.grid_search_hyperparameters(
                n_topics={"min": 10, "max": 50, "step": 10},
                alpha={"min": 0.01", "max": 0.5", "step": 0.1},
                beta={"min": 0.01", "max": 0.5", "step": 0.1}
            )
        """
        grid_results = {"Topics": [], "Alpha": [], "Beta": [], "Coherence": []}

        topics_range = list(np.arange(n_topics["min"], n_topics["max"], n_topics["step"]))
        alpha_values = list(np.arange(alpha["min"], alpha["max"], alpha["step"]))
        alpha_values.append("symmetric")
        alpha_values.append("asymmetric")

        beta_values = list(np.arange(beta["min"], beta["max"], beta["step"]))
        beta_values.append("symmetric")

        pbar = tqdm.tqdm(total=(len(beta_values) * len(alpha_values) * len(topics_range)))

        # Iterate through number of topics
        for k in topics_range:
            # Iterate through alpha values
            for a in alpha_values:
                # Iterate through beta values
                for b in beta_values:
                    pbar.set_description(f"{k=}, {a=}, {b=}")

                    # Fit the model and calculate coherence score
                    self.fit(num_topics=k, alpha=a, eta=b)
                    coherence_score = self.calculate_coherence_score(num_topics=k, alpha=a, eta=b)

                    # Save the grid search results
                    grid_results["Topics"].append(k)
                    grid_results["Alpha"].append(a)
                    grid_results["Beta"].append(b)
                    grid_results["Coherence"].append(coherence_score)

                    pbar.update(1)

        pbar.close()

        # Convert grid_results to DataFrame
        self.df_grid_results = pd.DataFrame(grid_results).sort_values(by="Coherence", ascending=False)

        # Select the row with the highest coherence score
        best_params = self.df_grid_results.loc[self.df_grid_results["Coherence"].idxmax()]

        # Assign the maximizing values to instance variables
        self.best_num_topics = best_params["Topics"]
        self.best_alpha = best_params["Alpha"]
        self.best_beta = best_params["Beta"]
