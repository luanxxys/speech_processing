# kaldi yesno example

yesno 实例是关于 yes 和 no 两个孤立词的识别。

识别器的任务是在**语音向量序列**和**隐藏的符号序列**间实现一个映射。

- ### 样本集介绍

    数据集中总共 60 个 wav 文件, 采样率都是 8k, wav 文件里每一个单词要么”ken” 要么”lo”(“yes” 和”no”) 的发音, 所以每个文件有 8 个发音, 文件命名中的 1 代表 yes 发音, 0 代表 no 的发音。

- ### 脚本文件介绍

    所需脚本文件位于 `kaldi/egs/yesno/s5` 中。

    + conf 文件夹里是一些配置文件例如 MFCC 的参数 HMM 的拓扑结构
    + local 文件夹里主要是一些准备数据的脚本 供顶层脚本 run.sh 调用
    + steps 和 utils 文件夹里主要是一些运行时调用的脚本
    + data 文件夹里主要存放语言模型、发音字典和音素信息等

- ### steps

    + 下载数据集 waves_yesno.tar.gz

        http://www.openslr.org/resources/1/waves_yesno.tar.gz

        或者

        http://sourceforge.net/projects/kaldi/files/waves_yesno.tar.gz

    + 解压缩到 `path/to/kaldi/egs/yesno/s5` 目录下

    + 查看、确认、运行 `run.sh`

- ### run.sh 解构

    ```bash
    #!/bin/bash

    train_cmd="utils/run.pl"
    decode_cmd="utils/run.pl"

    # kaldi/egs/yesno/s5 目录下没有 waves_yesno.tar.gz 文件, 则要下载该文件
    if [ ! -d waves_yesno ]; then
      wget http://www.openslr.org/resources/1/waves_yesno.tar.gz || exit 1;
      # was:
      # wget http://sourceforge.net/projects/kaldi/files/waves_yesno.tar.gz || exit 1;
      # 解压后, waves_yesno 文件夹下的文件如下
      # 0_0_0_0_1_1_1_1.wav
      # ...
      # 1_1_1_0_1_0_1_1.wav
      tar -xvzf waves_yesno.tar.gz || exit 1;
    fi

    train_yesno=train_yesno
    test_base_name=test_yesno

    rm -rf data exp mfcc

    # Data preparation

    local/prepare_data.sh waves_yesno

        # 生成 wavelist 文件：把 waves_yeno 目录下的文件名全部保存到 waves_all.list 中
        ls -1 $waves_dir > data/local/waves_all.list
        # 将 waves_all.list 中的 60 个 wav 文件, 分成两拨，各 30 个，分别放在 waves.test 和 waves.train 中
        local/create_yesno_waves_test_train.pl waves_all.list waves.test waves.train
        # 根据 waves.test 和 waves.train 又会生成 test_yesno_wav.scp 和 train_yesno_wav.scp 两个文件
        # 其中用于训练的 scp 文件如下
            - 生成 train_yesno.txt 和 test_yesno.txt， 这两个文件存放的是发音 id 和对应的文本
            - 生成 utt2spk 和 spk2utt 个文件分别是发音和人对应关系，以及人和其发音 id 的对应关系．由于只有一个人的发音，所以这里都用 global 来表示发音
        # 此外还可能会有如下文件（这个例子没有用到）
            - segments
                包括每个录音的发音分段 / 对齐信息（只有在一个文件包括多个发音时需要）
            - reco2file_and_channel
                双声道录音情况使用到
            - spk2gender
                将说话人和其性别建立映射关系，用于声道长度归一化
        # 目录结构
            data
            ├───train_yesno
            │   ├───text
            │   ├───utt2spk
            │   ├───spk2utt
            │   └───wav.scp
            └───test_yesno
                ├───text
                ├───utt2spk
                ├───spk2utt
                └───wav.scp

    # 构建语言学知识－词汇和发音词典．需要用到 steps 和 utils 目录下的工具。这可以通过修改该目录下的 path.sh 文件进行更新。
    local/prepare_dict.sh
        # 首先创建词典目录
        mkdir -p data/local/dict
        # 这个简单的例子只有两个单词：YES 和 NO，为简单起见，这里假设这两个单词都只有一个发音：Y 和 N。这个例子直接拷贝了相关的文件，非语言学的发音，被定义为 SIL
        data/local/dict/lexicon.txt
        <SIL> SIL
        YES Y
        NO N

        lexicon.txt，完整的词位 - 发音对
        lexicon_words.txt，单词 - 发音对
        silence_phones.txt， 非语言学发音
        nonsilence_phones.txt，语言学发音
        optional_silence.txt ，备选非语言发音
        # 把字典转换成 kaldi 可以接受的数据结构 - FST（finit state transducer）
        utils/prepare_lang.sh --position-dependent-phones false data/local/dict "<SIL>" data/local/lang data/lang

    # 由于语料有限，所以将位置相关的发音 disable
    utils/prepare_lang.sh --position-dependent-phones false data/local/dict "<SIL>" data/local/lang data/lang
    local/prepare_lm.sh
    # OOV 存放的是词汇表以外的词，这里就是静音词（非语言学发声意义的词），发音字典是二进制的 OpenFst 格式
    # 语言学模型
        这里使用的是一元文法语言模型，同样要转换成 FST 以便 kaldi 接受。该语言模型原始文件是 data/local/lm_tg.arpa，生成好的 FST 格式的。是字符串和整型值之间的映射关系，kaldi 里使用整型值。
        # 查看生成音素的树形结构
        ~/kaldi/src/bin/draw-tree data/lang/phones.txt exp/mono0a/tree | dot -Tps -Gsize=8,10.5 | ps2pdf - ./tree.pdf
        # LM（language model）在 data/lang_test_tg/
        local/prepare_lm.sh
    # 转移模型
    # 音素 hmm 状态
    # 高斯模型
    # 编译训练图
        为每一个训练的发音编译 FST，为训练的发句编码 HMM 结构。
    ```

    ```bash
    # Feature extraction
    for x in train_yesno test_yesno; do
     steps/make_mfcc.sh --nj 1 data/$x exp/make_mfcc/$x mfcc
     steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc
     utils/fix_data_dir.sh data/$x
    done

    # Mono training
    steps/train_mono.sh --nj 1 --cmd "$train_cmd" \
      --totgauss 400 \
      data/train_yesno data/lang exp/mono0a

    # Graph compilation
    utils/mkgraph.sh data/lang_test_tg exp/mono0a exp/mono0a/graph_tgpr

    # Decoding
    steps/decode.sh --nj 1 --cmd "$decode_cmd" \
        exp/mono0a/graph_tgpr data/test_yesno exp/mono0a/decode_test_yesno

    for x in exp/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
    ```
