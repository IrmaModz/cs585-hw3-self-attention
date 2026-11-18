# CS585 HW3: Simplified Self-Attention

This project demonstrates a simplified sequential self-attention mechanism using
pretrained word embeddings. It is based on `CS585_P03_A20597307.py`.

The program:

1. Downloads and loads the Google News Word2Vec model (`word2vec-google-news-300`)
   through Gensim.
2. Prompts for an English sentence.
3. Splits the sentence by whitespace into tokens.
4. Checks that every token has a pretrained embedding.
5. Asks the user to select a token to analyze or exit.
6. Fits a PCA projection over 10,000 pretrained vectors and uses the first 30
   components (10% of the 300-dimensional embedding size) as query and key
   vectors.
7. Uses the original 300-dimensional word embeddings as value vectors.
8. Computes scaled dot-product self-attention for each token:

   ```text
   score(i, j) = dot(Q[i], K[j]) / sqrt(d_k)
   alpha(i, j) = softmax(score(i, j))
   Z[i] = sum(alpha(i, j) * V[j])
   ```

9. Finds three static nearest neighbors of the selected word.
10. Finds three nearest neighbors of the selected word's contextualized embedding
    in the static Word2Vec vocabulary.
11. Projects the target, contextualized target, and both neighbor groups to 3D
    and displays an interactive Matplotlib 3D plot.

There is no learned or randomly initialized query, key, or value weight matrix:
PCA is used for queries and keys, and values are identity-mapped to the static
Word2Vec vectors.

## Requirements

- Python 3
- NumPy
- Gensim
- Matplotlib
- scikit-learn

Install the packages with:

```bash
python -m pip install numpy gensim matplotlib scikit-learn
```

## Run

From the project directory:

```bash
python CS585_P03_A20597307.py
```

On the first run, Gensim downloads the Google News vectors. This model is large
(approximately 1.6 GB), and the download may take some time.

Example session:

```text
Please enter a sentence: weather is nice
Input sentence: weather is nice

Which word do you want to analyze?
[1] weather
[2] is
[3] nice
[4] EXIT
Enter a number: 1
```

A 3D plot then opens. A Matplotlib window must remain available for the plot to
display.

## Input limitations

- Tokenization is whitespace-based. Punctuation attached to a token is preserved
  as part of that token.
- The sentence may have any number of words, but every token must exist exactly
  in the Word2Vec vocabulary.
- If a token is out of vocabulary, the program exits with:

  ```text
  Embeddings for some tokens are not available. Run the program again and try a different sentence
  ```
