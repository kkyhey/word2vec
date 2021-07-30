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
def one_hot(word):
    one_hot_vec = np.zeros(len(cv.vocabulary_))
    one_hot_vec[cv.vocabulary_[word]] = 1
    return one_hot_vec


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

# Skip-gram
np.random.seed(42)

n_vocab = len(cv.vocabulary_)
epoch = 10000
window_size = 1
n_hidden = 6
lr = 1e-2
threshold = 1e-8
interval = 100

words_list = [text.split("\u3000") for text in corpus]
sg_weight = 1e-2*np.random.randn(n_vocab, n_hidden)
sg_bias = 1e-2*np.random.randn(1, n_hidden)
out_weight = 1e-2*np.random.randn(n_hidden, n_vocab)
out_bias = 1e-2*np.random.randn(1, n_vocab)

prev_loss = 0.
losses = []
for words in words_list:
    indices = list(range(len(words)))
    np.random.shuffle(indices)
    for i in indices:
        # Ready inputs
        #論文に基づきRを区間[1,C]からランダムに選択。CBoWとは逆に入力が1単語、正解ラベルが複数単語
        #正解ラベルの数だけ予測と学習を繰り返す
        R = np.random.randint(1, window_size, 1)[0] \
            if window_size > 1 else 1
        input = words[i]
        ans = get_surround(words, i, R)
        x = np.array(one_hot(input)).reshape(1, -1)
        y = np.array([one_hot(a) for a in ans])

        for j in range(len(y)):
            # Forward
            hidden = (x@sg_weight + sg_bias).mean(axis=0, keepdims=True)
            output = hidden@out_weight + out_bias
            predict = np.exp(output)/np.sum(np.exp(output), keepdims=True)
    
            # Backward
            delta = predict - y[j]
            d_out_weight = hidden.T@delta
            d_out_bias = delta.copy()
            d_hidden = delta@out_weight.T
            d_hidden = np.tile(d_hidden, x.shape[0])\
                        .reshape(x.shape[0], -1)*x.shape[0]
            d_hidden_weight = x.T@d_hidden
            d_hidden_bias = np.sum(d_hidden, axis=0, keepdims=True)

            # Update
            sg_weight -= lr*d_hidden_weight
            sg_bias -= lr*d_hidden_bias
            out_weight -= lr*d_out_weight
            out_bias -= lr*d_out_bias
print(sg_weight)


#推論結果
#入力単語の周辺語に対応する確率がしっかり高くなることを確認
#周辺の候補単語が複数ある場合はそれらにも高めの確率が割り振られる
vocab = sorted(cv.vocabulary_, key=lambda x: x[0])
for w in vocab:
    print(f"入力: {w}")
    x = np.array(one_hot(w)).reshape(1, -1)

    hidden = (x@sg_weight + sg_bias).mean(axis=0, keepdims=True)
    output = hidden@out_weight + out_bias
    predict = np.exp(output)/np.sum(np.exp(output), keepdims=True)
    pprint(dict(zip(vocab, predict[0])))
    print("-----")

#単語間の類似度
sim1 = cosine_similarity(u[:, :6]@np.diag(s[:6]))
sim2 = cosine_similarity(sg_weight)
for i in range(n_vocab):
    for j in range(i+1, n_vocab):
        print(f"|{id2w[i]}|{id2w[j]}|{sim1[i, j]}|{sim2[i, j]}|")

#BowとCBowとskipgramの比較
sim1 = cosine_similarity(u[:, :6]@np.diag(s[:6]))
sim2 = cosine_similarity(cbow_weight)
sim3 = cosine_similarity(sg_weight)
for i in range(n_vocab):
    for j in range(i+1, n_vocab):
        print(f"|{id2w[i]}|{id2w[j]}|{sim1[i, j]}|{sim2[i, j]}|{sim3[i, j]}|")