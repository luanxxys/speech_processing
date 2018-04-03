# Kaldi

kaldi是一款基于c++编写的完全开源的语音识别工具箱。

- ## [Installation](installation.md)

- ## Kaldi toolbox info

    + ### 特点及优势

        * 支持主流的特征提取：MFCC PLP等
        * 支持传统的GMM-HMM的声学模型构建
        * 支持WFST的解码策略
        * 支持深度神经网络的声学建模
        * 完善的社区支持

    + ### 组织架构

        ![](images/frame_structure.png)

        * 外部库
        * kaldi自身的C++库
        * C++可执行文件
        * Shell脚本调用

    + ### 文件夹说明

        kaldi-trunk/tools目录下主要是一些外部库

        * 具体样例中

            - conf文件夹里是一些配置文件例如MFCC的参数 HMM的拓扑结构
            - local文件夹里主要是一些准备数据的脚本 供顶层脚本run.sh调用
            - steps和utils文件夹里主要是一些运行时调用的脚本
            - data文件夹里主要存放语言模型、发音字典和音素信息等等。




