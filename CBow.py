from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

corpus = ["私　は　ラーメン　が　好き　だ　。",
          "私　は　寿司　が　好き　だ　。",
          "彼　は　ラーメン　が　好き　だ　。"]

def tokenizer(text):
    return text.split("　")

cv = CountVectorizer(tokenizer=tokenizer)
weights = cv.fit_transform(corpus)
print(cv.get_feature_names())
print(weights.toarray())

#corpusのサイズが小さいためwindow_size = 1(論文では10)
#one_hotはある単語についてのone-hotベクトル表現を得る関数
def one_hot(word):
    one_hot_vec = np.zeros(len(cv.vocabulary_))
    one_hot_vec[cv.vocabulary_[word]] = 1
    return one_hot_vec

#あるターゲット単語(pointerで指示される)の周辺2*window_sizeの単語を取得する関数
def get_surround(words, pointer, window_size):
    input = words[max(pointer-window_size, 0)
                : min(pointer+window_size+1, len(words))]
    if pointer-window_size < 0:
        del input[pointer]
    elif pointer+window_size+1 > len(words):
        del input[pointer+window_size+1-len(words)]
    else:
        del input[window_size]
    return input

# CBoW

np.random.seed(42)

n_vocab = len(cv.vocabulary_)
epoch = 10000
window_size = 1
n_hidden = 6
lr = 1e-2
threshold = 1e-8
interval = 100

#コーパスを単語に分割したwords_listとWinとWoutを定義
words_list = [text.split("\u3000") for text in corpus]
cbow_weight = 1e-2*np.random.randn(n_vocab, n_hidden)
cbow_bias = 1e-2*np.random.randn(1, n_hidden)
out_weight = 1e-2*np.random.randn(n_hidden, n_vocab)
out_bias = 1e-2*np.random.randn(1, n_vocab)

#単語リストのアクセス順をランダムにシャッフル(過学習を防ぐため)
for words in words_list:
    indices = list(range(len(words)))
    np.random.shuffle(indices)
    for i in indices:
        # Ready inputs
        #get_surround関数でターゲット単語ansの周辺単語を取得し、xとyにそれぞれのone-hotベクトルを保持
        input = get_surround(words, i, window_size)
        ans = words[i]
        x = np.array([one_hot(word) for word in input])
        y = np.array(one_hot(ans)).reshape(1, -1)

        # Forward
        hidden = (x@cbow_weight + cbow_bias).mean(axis=0, keepdims=True)
        output = hidden@out_weight + out_bias
        predict = np.exp(output)/np.sum(np.exp(output), keepdims=True)

        # Backward
        delta = predict - y
        d_out_weight = hidden.T@delta
        d_out_bias = delta.copy()
        d_hidden = delta@out_weight.T
        d_hidden = np.tile(d_hidden, x.shape[0])\
                     .reshape(x.shape[0], -1)*x.shape[0]
        d_hidden_weight = x.T@d_hidden
        d_hidden_bias = np.sum(d_hidden, axis=0, keepdims=True)

        # Update
        #SGDでパラメータの更新
        cbow_weight -= lr*d_hidden_weight
        cbow_bias -= lr*d_hidden_bias
        out_weight -= lr*d_out_weight
        out_bias -= lr*d_out_bias
    
print(cbow_weight)

#推論結果
vocab = sorted(cv.vocabulary_, key=lambda x: x[0])
for words in words_list:
    indices = list(range(len(words)))
    for i in indices:
        # Ready inputs
        input = get_surround(words, i, window_size)
        ans = words[i]
        x = np.array([one_hot(word) for word in input])
        y = np.array(one_hot(ans)).reshape(1, -1)
        print(f"入力: {input}\t正解: {ans}")

        # Forward
        hidden = (x@cbow_weight + cbow_bias).mean(axis=0, keepdims=True)
        output = hidden@out_weight + out_bias
        predict = np.exp(output)/np.sum(np.exp(output), keepdims=True)
        pprint(dict(zip(vocab, predict[0])))
        print("-----")

#単語間の類似度
sim1 = cosine_similarity(u[:, :6]@np.diag(s[:6]))
sim2 = cosine_similarity(cbow_weight)
for i in range(n_vocab):
    for j in range(i+1, n_vocab):
        print(f"|{id2w[i]}|{id2w[j]}|{sim1[i, j]}|{sim2[i, j]}|")
