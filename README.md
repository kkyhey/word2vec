# word2vec  
BowとWord2vecを用いたCBowとSkip-gramの実装  

# この技術の背景  
- コンピュータに自然言語を処理させるために単語を数値化(ベクトル化)する必要があり、このベクトルを分散表現と呼ぶ  
- 数値に変換する際は可能な限り元の単語の復元が可能であること  
- ベクトルの次元数が大きいほど計算コストが高くなるが、代わりに変換前の単語の表現力が高い  

# Bow  
- BoW: Bag of Wordsは単語の出現頻度に着目して分散表現を作成する方法(単語が出現していれば1、そうでなければ0)  
- 分散表現を得る基本的な手法だが、単語数に比例してベクトルの次元数が増えるため計算量が大きくなる  
- また、単語の前後関係や共起によって分散表現を作成するため単語のニュアンスの違いまで考慮できていない  

# CBow  
- CBoW: Continuous Bag of Wordsはある文章の特定の1単語を隠し、文章から単語を推測するというタスクを解くニューラルネットワークから作成されるWord2Vec分散表現  
- BoWに比べ、0≤cosθ≤1⇔0°≤θ≤90°0≤cos⁡θ≤1⇔0°≤θ≤90° だった表現範囲が−1≤cosθ≤1⇔0°≤θ≤180°−1≤cos⁡θ≤1⇔0°≤θ≤180°に広がっている  

# Skip-gram  
- skip-gramはCBoWとは逆にある文章の特定の1単語に注目し、単語から文章内の周辺語を推測するというタスクを解くニューラルネットワークから作成されるWord2Vec分散表現  

# 評価  
- CBoWよりもskip-gramで生成された分散表現の方が単語の概念的により的確なものである場合が多い。ただし学習コストはskip-gramの方が高くなる  
| 単語1    | 単語2    | 圧縮後の類似度 | CBoWの類似度 | skip-gramの類似度 |  
|----------|----------|----------------|--------------|-------------------|  
| 。       | が       | 0              | 0.60990927   | 0.22291748        |  
| 。       | だ       | 0              | 0.00915309   | -0.36743359       |  
| 。       | は       | 0              | -0.07209241  | 0.27237642        |  
| 。       | ラーメン | -9.36E-16      | -0.82747653  | -0.47548472       |  
| 。       | 好き     | 0.70710678     | 0.41772277   | 0.78170257        |  
| 。       | 寿司     | -8.31E-16      | -0.8332996   | -0.45655293       |  
| 。       | 彼       | -3.81E-16      | -0.77846921  | -0.84750624       |  
| 。       | 私       | -3.81E-16      | -0.81165613  | -0.84751732       |  
| が       | だ       | 0.56694671     | 0.56699832   | 0.36404029        |  
| が       | は       | 0.42257713     | -0.36296794  | 0.33974349        |  
| が       | ラーメン | 0              | -0.92665413  | -0.8376686        |  
| が       | 好き     | 0              | -0.3416646   | -0.10645191       |  
| が       | 寿司     | 0              | -0.92996123  | -0.85372539       |  
| が       | 彼       | 0              | -0.40925639  | -0.46191969       |  
| が       | 私       | 0              | -0.53200333  | -0.46796418       |  
| だ       | は       | 5.89E-17       | -0.87007966  | -0.5124081        |  
| だ       | ラーメン | 0              | -0.47367427  | -0.09785801       |  
| だ       | 好き     | 0              | -0.43023514  | -0.16833614       |  
| だ       | 寿司     | 0              | -0.44468693  | -0.1128717        |  
| だ       | 彼       | 0              | -0.16763127  | -0.04022722       |  
| だ       | 私       | 0              | -0.25452483  | -0.04343022       |  
| は       | ラーメン | 0              | 0.33401352   | -0.6014627        |  
| は       | 好き     | 0              | 0.10352886   | -0.19527698       |  
| は       | 寿司     | 0              | 0.29784482   | -0.61471978       |  
| は       | 彼       | 0              | 0.26516785   | -0.14841524       |  
| は       | 私       | 0              | 0.32057734   | -0.14655646       |  
| ラーメン | 好き     | 0.5            | 0.02933748   | 0.0651404         |  
| ラーメン | 寿司     | 1              | 0.99689137   | 0.99848701        |  
| ラーメン | 彼       | 0.70710678     | 0.67113196   | 0.48667692        |  
| ラーメン | 私       | 0.70710678     | 0.76239648   | 0.4911722         |  
| 好き     | 寿司     | 0.5            | 0.06470027   | 0.07169858        |  
| 好き     | 彼       | -1.68E-16      | -0.67816785  | -0.80879557       |  
| 好き     | 私       | -1.68E-16      | -0.57679278  | -0.80612887       |  
| 寿司     | 彼       | 0.70710678     | 0.62734895   | 0.49199579        |  
| 寿司     | 私       | 0.70710678     | 0.72169384   | 0.49631656        |  
| 彼       | 私       | 1              | 0.99004629   | 0.99995452        |  
