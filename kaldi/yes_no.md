# kaldi yesno example

yesno 实例是关于 yes 和 no 两个孤立词的识别。

- 样本集介绍

    数据集中总共 60 个 wav 文件, 采样率都是 8k, wav 文件里每一个单词要么”ken” 要么”lo”(“yes” 和”no”) 的发音, 所以每个文件有 8 个发音, 文件命名中的 1 代表 yes 发音, 0 代表 no 的发音。

所需脚本文件位于 `kaldi/egs/yesno/s5` 中。

- conf 文件夹里是一些配置文件例如 MFCC 的参数 HMM 的拓扑结构
- local 文件夹里主要是一些准备数据的脚本 供顶层脚本 run.sh 调用
- steps 和 utils 文件夹里主要是一些运行时调用的脚本
- data 文件夹里主要存放语言模型、发音字典和音素信息等等

## steps

- 下载 waves_yesno.tar.gz

    http://www.openslr.org/resources/1/waves_yesno.tar.gz

    或者

    http://sourceforge.net/projects/kaldi/files/waves_yesno.tar.gz

- 解压缩到 `path/to/kaldi/egs/yesno/s5` 目录下

- 查看、确认 `run.sh` 脚本

## theory

[kaldi yesno example](https://blog.csdn.net/shichaog/article/details/73264152)
