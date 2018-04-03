# Installing Kaldi

[Github site](https://github.com/kaldi-asr/kaldi/)

- ### Clone repo

        $ git clone https://github.com/kaldi-asr/kaldi

- ### Compile

     Go to the below directorys one by one and follow each INSTALL instructions there.

        $ cd tools
        $ extras/check_dependencies.sh
        $ make -j 16

        $ cd ../src
        $ ./configure --shared
        $ make depend -j 16
        $ make -j 16

    > 16 >>> Number of CPU
