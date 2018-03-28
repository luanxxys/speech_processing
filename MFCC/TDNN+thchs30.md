# Use TDNN model to train thchs30 set

### preparation

- [THCHS-30 : A Free Chinese Speech Corpus](https://arxiv.org/abs/1512.01882)
- [thchs30 download](http://www.openslr.org/18/)
- [CVTE 公司开源其训练好的 TDNN 模型](http://kaldi-asr.org/models.html)
- [thchs30 scripts dataset(kaldi-asr)](https://github.com/kaldi-asr/kaldi/tree/master/egs/thchs30)

### prediction

利用 kaldi 中的 online2-wav-nnet3-latgen-faster 脚本,调用 thchs30 中相应的文件,识别出 thchs30 中的 wav 文件

    ./data/luanxiaoyang/speech_processing/kaldi/src/online2bin/online2-wav-nnet3-latgen-faster --do-endpointing=false --online=false --feature-type=fbank --fbank-config=/data/luanxiaoyang/speech_processing/database/CVTE\ Mandarin\ Model/cvte/s5/conf/fbank.conf --max-active=7000 --beam=15.0 --lattice-beam=6.0 --acoustic-scale=1.0 --word-symbol-table=/data/luanxiaoyang/speech_processing/database/CVTE\ Mandarin\ Model/cvte/s5/exp/chain/tdnn/graph/words.txt /data/luanxiaoyang/speech_processing/database/CVTE\ Mandarin\ Model/cvte/s5/exp/chain/tdnn/final.mdl /data/luanxiaoyang/speech_processing/database/CVTE\ Mandarin\ Model/cvte/s5/exp/chain/tdnn/graph/HCLG.fst 'ark:echo utter1 utter1|' 'scp:echo utter1 /data/luanxiaoyang/speech_processing/database/CVTE\ Mandarin\ Model/cvte/s5/data/wav/00030/2017_03_07_16.57.22_1175.wav|' ark:/dev/null

### Reference

- [Time Delay Nerual Networks](https://zhuanlan.zhihu.com/p/28283265)
- [TDNN时延神经网络](http://blog.csdn.net/richard2357/article/details/16896837)


- 待解决

    判别结果显示(wer ...)
