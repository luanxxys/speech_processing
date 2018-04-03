# voicebox

- ## Inside function info

    > 以下内容可在 用`help voicebox`命令查看

    + ##### 音频文件输入或输出

            readwav       - 读取 WAV 文件
            writewav      - 写 WAV 文件
            readhtk       - 读 HTK waveform 文件
            writehtk      - 写 HTK waveform 文件
            readsfs       - 读 SFS 文件
            readsph       - 读 SPHERE/TIMIT waveform 文件
            readaif       - 读 AIFF Audio Interchange file format 文件
            readcnx       - 读 BT Connex database 文件
            readau        - 读 AU 文件 (from SUN)
            readflac      - 读 FLAC 文件

    + ##### 频率尺度转换

            frq2bark      - Convert Hz to the Bark frequency scale 利用基本频率 hz 转换到 Bark 频率尺度
            frq2cent      - Convert Hertz to cents scale 利用基本频率 hz 转换到 cents 尺度
            frq2erb       - Convert Hertz to erb rate scale 利用基本频率 hz 转换到 erb 比例尺度
            frq2mel       - Convert Hertz to mel scale 利用基本频率 hz 转换到梅尔尺度
            frq2midi      - Convert Hertz to midi scale of semitones 利用基本频率 hz 转换到 MIDI 文件音高
            bark2frq      - Convert the Bark frequency scale to Hz 利用 Bark 频率尺度转换到基本频率 hz
            cent2frq      - Convert cents scale to Hertz 利用 cents 尺度转换到基本频率 hz
            erb2frq       - Convert erb rate scale to Hertz 利用 erb 比尺度转换到基本频率 hz
            mel2frq       - Convert mel scale to Hertz 利用梅尔尺度转换高基本频率 hz
            midi2frq      - Convert midi scale of semitones to Hertz 利用 midi 文件音高转换到基本频率 hz

    + ##### 傅里叶 Fourier / 离散余弦 DCT / 离散哈脱莱 Hartley 变换

            rfft          - FFT of real data 实数的傅里叶变换
            irfft         - Inverse of FFT of real data 实数的反傅里叶变换
            rsfft         - FFT of real symmetric data 实对称数据的傅里叶变换
            rdct          - DCT of real data 实数的离散余弦变换
            irdct         - Inverse of DCT of real data 实数的反离散余弦变换
            rhartley      - Hartley transform of real data 实数的离散哈脱莱变换
            zoomfft       - calculate the fft over a portion of the spectrum with any resolution 任意分辨率的频谱傅里叶计算变换
            sphrharm      - calculate forward and inverse shperical harmonic transformations 正向和反向球面谐波计算变换

    + ##### Speech Recognition 语音识别

            melbankm      - Mel filterbank transformation matrix 梅尔滤波器组变换矩阵
            melcepst      - Mel cepstrum frontend for recogniser 梅尔倒频谱前端识别
            cep2pow       - Convert mel cepstram means & variances to power domain 利用梅尔倒频谱均值和方差转换到功率域
            pow2cep       - Convert power domain means & variances to mel cepstrum 利用功率域转换到梅尔倒频谱均值和方差
            ldatrace      - constrained Linear Discriminant Analysis to maximize trace(W\B) 约束线性分析到最大限度跟踪

    + ##### Speech Analysis 语音分析
    + ##### Speech Synthesis 语音合成
    + ##### Probability Distributions 概率分布
    + ##### Vector Distances 向量距离
    + ##### LPC Analysis of Speech 语音线性功能控制器 LPC 分析
    + ##### Speech Coding 语音编码
    + ##### Signal Processing 信号处理
    + ##### Information Theory 信息理论
    + ##### Computer Vision 文本计算
    + ##### Printing and Display functions 打印展示函数
    + ##### ... ...

- ## Installation

    + ### [Download](http://www.ee.ic.ac.uk/hp/staff/dmb/voicebox/voicebox.zip)

    + ### Add to Matlab

        1. 解压 voicebox.zip，将整个目录 voicebox 复制到 MATLAB 的安装目录下, 如

                /usr/local/MATLAB/R2017a/toolbox
                C:\Program Files\MATLAB\R2017a\toolbox

        1. 找到 'pathdef.m' 文件，打开，把上面路径添加到该文件中，保存。

                matlabroot,'\toolbox\voicebox;', ...

                linux 下 ; ---> :

            > C:\Program Files\MATLAB\R2017a\toolbox\local\pathdef.m
            >
            > /usr/local/MATLAB/R2017b/toolbox/local/pathdef.m

        1. 运行 `rehash toolboxcache` 命令，完成工具箱加载

        1. 测试 `what voicebox`
