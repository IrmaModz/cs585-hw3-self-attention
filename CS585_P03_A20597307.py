import random
import re
import sys
from matplotlib import text
import gensim.downloader as gsd
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
    print(f"Missing tokens: {missing}")  
    print("Embeddings for some tokens are not available. "
          "Run the program again and try a different sentence")
    sys.exit()
# ----------------------------
# 4. If all static embeddings exist, ask the user to pick one of the words (WORD) in the sentence, for example:
    # Input sentence: weather is nice 

    # Which word do you want to analyze?
        #[1] weather
        #[2] is
        #[3] nice
        #[4] EXIT
# -------------------------
# 5. and create all contextualized word embeddings using a simplified self-attention mechanism. Simplifications:
    # your self-attention will run sequentially (real one is parallelized),
    # there is NO query weight matrix. Use a PCA or t-SNE projection/dimensionality reduction approach instead (you can assume query dimension to be ~10% of the actual static embedding dimension; it's fine to use existing Python packages for that purpose),
    # there is NO key weight matrix. Use a PCA or t-SNE (same as for query generation) projection/dimensionality reduction approach instead (you can assume key dimension to be ~10% of the actual static embedding dimension; it's fine to use existing Python packages for that purpose),
    # there is NO value weight matrix or it is an identity matrix. In other words: value vector is the same as the input static embedding.
# --------------------------
# 6. for the static embedding of WORD find its 3 (static WORD embedding static neighbors) nearest (cosine similarity is fine or you can use Euclidean distance) neighbors in the static embedding space,
# --------------------------
# 7. for the contextualized embedding of WORD find its 3 (contextualized WORD  embedding static neighbors) nearest (cosine similarity is fine or you can use Euclidean distance) neighbors in the static embedding space,

# ---------------------------
# 8. plot in 3D (use different color and / or labels) on a single figure:
    # static WORD embedding,
    # new, contextualized, WORD embedding,
    # static WORD embedding static nearest neighbors,
    # contextualized WORD embedding static nearest neighbors,
    # you can use PCA or t-SNE methods to reduce dimensionality (it's fine to use existing Python packages for that purpose).



# For demonstration: pick two random words from the vocabulary
# For simplicity, we look only at a subset of lowercased words
# made ofalphabetic tokens to avoid phrases, numbers, and cases like "Wal_Mart"
word_candidates = [w for w in model.index_to_key if w.isalpha() and w.islower()]
random_word1, random_word2 = random.sample(word_candidates, 2)

# Cosine similarity between the two word vectors
similarity = model.similarity(random_word1, random_word2)

print(f"Words: '{random_word1}' and '{random_word2}'")
print(f"Cosine similarity: {similarity:.4f}")
