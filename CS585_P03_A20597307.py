import random
import re
import sys
from matplotlib import pyplot as plt, text
import gensim.downloader as gsd
import numpy as np
from sklearn.decomposition import PCA
# You may need to install gensim first with: pip install gensim

# Download and load the Google News word2vec vectors
model = gsd.load("word2vec-google-news-300")

# alternative sets: "glove-wiki-gigaword-300" (GloVe) or "fasttext-wiki-news-subwords-300"

# -------------------------
# 1. Tokenize the sentence using the white-space-based tokenization [you are welcome to go beyond that and change it to BPE (your own implementation or existing code. Keep in mind the relationship between BPE vocabulary and your embeddings set!) for more self-exploration, BUT your submission should use white-space-based tokenization] approach.
def tokenize_sentence(sentence):
    # Simple whitespace-based tokenization
    tokens = sentence.split()
    #V = sorted(set(tokens))
    #word_to_index = {word: i for i, word in enumerate(V)}
    return tokens

# -------------------------
# 2. input a sentence from the user and tokenize it using the function above.
while True:
    user_sentence = input("Please enter a sentence: ").strip()
    if user_sentence:
        break
    print("Empty input - please type at least one word.")

tokens = tokenize_sentence(user_sentence)

# ----------------------------
# 3. Check if all the tokens are present in your word embeddings set. If not, display a note 
    # "Embeddings for some tokens are not available. Run the program again and try a different sentence" 
    # and exit [BPE will not have that problem, but that's beyond the scope of the assignment].
missing = [tok for tok in tokens if tok not in model]
if missing:
    print("Embeddings for some tokens are not available. Run the program again and try a different sentence")
    sys.exit()

# ----------------------------
# 4. If all static embeddings exist, ask the user to pick one of the words (WORD) in the sentence, for example:
    # Input sentence: weather is nice 

    # Which word do you want to analyze?
        #[1] weather
        #[2] is
        #[3] nice
        #[4] EXIT
if not missing:
    print("Input sentence:", user_sentence)
    print("\nWhich word do you want to analyze?")
    for i, tok in enumerate(tokens, start=1):
        print(f"[{i}] {tok}")
    print(f"[{len(tokens) + 1}] EXIT")

    while True:
        choice = input("Enter a number: ").strip()

        if choice.isdigit() and 1 <= int(choice) <= len(tokens) + 1:
            choice = int(choice)
            break
        print(f"Please enter a number between 1 and {len(tokens) + 1}.")

if choice == len(tokens) + 1:
        print("Exiting.")
        sys.exit()
 
word_idx = choice - 1
WORD = tokens[word_idx]
print(f"You selected: {WORD}")     
# -------------------------
# 5. and create all contextualized word embeddings using a simplified self-attention mechanism. Simplifications:
    # your self-attention will run sequentially (real one is parallelized),
    # there is NO query weight matrix. Use a PCA or t-SNE projection/dimensionality reduction approach instead (you can assume query dimension to be ~10% of the actual static embedding dimension; it's fine to use existing Python packages for that purpose),
    # there is NO key weight matrix. Use a PCA or t-SNE (same as for query generation) projection/dimensionality reduction approach instead (you can assume key dimension to be ~10% of the actual static embedding dimension; it's fine to use existing Python packages for that purpose),
    # there is NO value weight matrix or it is an identity matrix. In other words: value vector is the same as the input static embedding.
D   = model.vector_size
d_k = int(0.10 * D)


pca = PCA(n_components=d_k, whiten=False, random_state=0)
pca.fit(model.vectors[:10000])     # 10k most frequent words (see note 4)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

X = np.array([model[tok] for tok in tokens])
n = len(tokens)

Q = pca.transform(X)
K = Q
V = X

Z = np.zeros((n, D), dtype=np.float64)
attention = np.zeros((n, n))

for i in range(n):
    scores = np.array([np.dot(Q[i], K[j]) / np.sqrt(d_k) for j in range(n)])
    alpha = softmax(scores)
    attention[i] = alpha
    Z[i] = np.sum(alpha[:, np.newaxis] * V, axis=0)

z_word = Z[word_idx]

print(f"\nAttention weights for '{WORD}':")
for tok, a in zip(tokens, attention[word_idx]):
    print(f"  {tok}: {a:.4f}")
    
# --------------------------
# 6. for the static embedding of WORD find its 3 (static WORD embedding static neighbors) nearest (cosine similarity is fine or you can use Euclidean distance) neighbors in the static embedding space,
top3_static = model.most_similar(
    positive=[WORD],
    topn=3
)

print(f"\n3 nearest static neighbors of '{WORD}':")
for w, sim in top3_static:
    print(f"  {w}: {sim:.4f}")

# --------------------------
# 7. for the contextualized embedding of WORD find its 3 (contextualized WORD  embedding static neighbors) nearest (cosine similarity is fine or you can use Euclidean distance) neighbors in the static embedding space,

# word_candidates = set(w for w in model.index_to_key if w.isalpha() and w.islower())
top3_contextualized = model.most_similar(
    positive=[z_word],
    topn=3
)

print(f"\n3 nearest neighbors of contextualized '{WORD}':")
for w, sim in top3_contextualized:
    print(f"  {w}: {sim:.4f}")

# ---------------------------
# 8. plot in 3D (use different color and / or labels) on a single figure:
    # static WORD embedding,
    # new, contextualized, WORD embedding,
    # static WORD embedding static nearest neighbors,
    # contextualized WORD embedding static nearest neighbors,
    # you can use PCA or t-SNE methods to reduce dimensionality (it's fine to use existing Python packages for that purpose).
