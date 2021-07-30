import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

corpus = ["私　は　ラーメン　が　好き　だ　。",
          "私　は　寿司　が　好き　だ　。",
          "彼　は　ラーメン　が　好き　だ　。"]

def tokenizer(text):
    return text.split("　")

#文章ベクトル作成
cv = CountVectorizer(tokenizer=tokenizer)
weights = cv.fit_transform(corpus)
print(cv.get_feature_names())
print(weights.toarray())

#cos類似度で3つの文章の類似度を計算

x = weights.toarray()
sim = cosine_similarity(x)
for i in range(len(x)):
    for j in range(i+1, len(x)):
        print(f"文章{i+1}と文章{j+1}の類似度：{sim[i, j]}")

#単語の共起行列作成
vocab = cv.get_feature_names()
n_vocab = len(vocab)
w2id = dict(zip(vocab, range(n_vocab)))
id2w = dict(zip(range(n_vocab), vocab))

window_size = 1
cooccurrence = np.zeros((n_vocab, n_vocab), dtype=int)
for text in corpus:
    split_text = tokenizer(text)
    for i, word in enumerate(split_text):
        for j in range(1, window_size+1):
            if i - j >= 0:
                cooccurrence[w2id[word], w2id[split_text[i-j]]] += 1
            if i + j < len(split_text):
                cooccurrence[w2id[word], w2id[split_text[i+j]]] += 1
cooccurrence

#単語間の類似度計算
sim = cosine_similarity(cooccurrence)
for i in range(n_vocab):
    for j in range(i+1, n_vocab):
        print(f"「{id2w[i]}」と「{id2w[j]}」の類似度：{sim[i, j]}")

#共起行列に対して特異値分解を行い次元削減
np.set_printoptions(linewidth=200)
u, s, vh = np.linalg.svd(cooccurrence, full_matrices=False)
print("----- u -----")
print(u)
print("----- s -----")
print(s)
print("----- vh -----")
print(vh)

#s行列の要素が大きい6番目までで次元圧縮
compress_coocurrence = u[:, :6]@np.diag(s[:6])@vh[:6, :]
print(compress_coocurrence)

#圧縮前後の2乗誤差を比較
print(np.linalg.norm(cooccurrence - compress_coocurrence))

#グラフで次元削減の効果を確認
diff = []
for i in range(1, n_vocab+1):
    compress_coocurrence = u[:, :i]@np.diag(s[:i])@vh[:i, :]
    diff.append(np.linalg.norm(cooccurrence - compress_coocurrence))

plt.title("SVD compression")
plt.yscale("log")
plt.plot(range(1, n_vocab+1), diff)
plt.grid()
plt.xlabel("compress order")
plt.ylabel("norm error")
plt.show()

#9×9の共起行列を9×6に削減
print(u[:, :6]@np.diag(s[:6]))

#単語間の類似度を計算
sim = cosine_similarity(u[:, :6]@np.diag(s[:6]))
for i in range(n_vocab):
    for j in range(i+1, n_vocab):
        print(f"「{id2w[i]}」と「{id2w[j]}」の類似度：{sim[i, j]}")

