# GMM-HMM SR Model

- [从朴素贝叶斯到隐马尔科夫模型](从朴素贝叶斯到隐马尔科夫模型.pdf)
- [viterbi](viterbi.pdf)
- [HMM模型+Viterbi算法实例](HMM模型+Viterbi算法实例.pdf)

## 待完善, 目前状态不能上架

## GMM

GMM 是神马？怎样用 GMM 求某一音素（phoneme）的概率？

简单理解混合高斯模型就是几个高斯的叠加。

GMM for state sequence

    每个 state 有一个 GMM，包含 k 个高斯模型参数。

    其中，每个 GMM 有一些参数，就是我们要 train 的输出概率参数。

    怎么求呢？和 KMeans 类似，如果已知每个点 x^n 属于某每类 j 的概率 p(j|x^n)，则可以估计其参数。

    只要已知了这些参数，我们就可以在 predict（识别）时在给定 input sequence 的情况下，计算出一串状态转移的概率。

## Hidden Markov Model

最开始时，我们指定 HMM 的结构，训练 HMM 模型时：给定 n 个时序信号 y1...yT（训练样本）, 用 MLE（typically implemented in EM） 估计参数

    N 个状态的初始概率
    状态转移概率 a
    输出概率 b

在语音处理中，一个 word 由若干 phoneme（音素）组成。

每个 HMM 对应于一个 word 或者音素（phoneme）。

一个 word 表示成若干 states，每个 state 表示为一个音素。

用 HMM 需要解决 3 个问题

- Likelihood: 一个 HMM 生成一串 observation 序列 x 的概率 <the Forward algorithm>
- Decoding: 给定一串 observation 序列 x，找出最可能从属的 HMM 状态序列 <the Viterbi algorithm>
- Training: 给定一个 observation 序列 x，训练出 HMM 参数λ = {aij, bij}  the EM (Forward-Backward) algorithm

## GMM+HMM 大法解决语音识别

- ### 识别

    我们获得 observation 是语音 waveform, 以下是一个词识别全过程：

    + 将 waveform 切成等长 frames，对每个 frame 提取特征（e.g. MFCC）
    + 对每个 frame 的特征跑 GMM，得到每个 frame(o_i) 属于每个状态的概率 b_state(o_i)
    + 根据每个单词的 HMM 状态转移概率 a 计算每个状态 sequence 生成该 frame 的概率; 哪个词的 HMM 序列跑出来概率最大，就判断这段语音属于该词

- ### 训练

    #### the main steps

    + 训练库的创建：词汇集中的每个元素进行多次录制，且与相应词汇做好标签
    + 声学分析：训练波形数据转换为一系列系数向量
    + 模型定义：为总词汇集中的每个元素定义一个HMM原型
    + 模型训练：使用训练数据对每个HMM模型进行初始化、训练
    + 任务定义：识别系统的语法（什么可被识别）的定义
    + 未知输入信号识别
    + 评估：识别系统的性能可通过测试数据进行评估

    上面说了怎么做识别。那么我们怎样训练这个模型以得到每个 GMM 的参数和 HMM 的转移概率什么的呢？

    + Training the params of GMM
    + Training the params of HMM

- [source](http://blog.csdn.net/abcjennifer/article/details/27346787)
