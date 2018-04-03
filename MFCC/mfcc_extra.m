%对输入的语音序列x进行mfcc参数提取，返回mfcc参数和一阶差分mfcc参数
%function ccc=mfcc(x);

%采样频率为fs=8000Hz,fh=4000Hz,语音信号的频率一般在300-3400Hz，所以一般情况下采样频率设为8000Hz即可。
[x fs]=wavread('test.wav');%使用 wav 文件自身的频率
M = 24;%mel三角滤波器的个数
N = 256;%语音片段（帧）的长度，也是fft的点数，也是m个滤波器(0-4000Hz)总点数，窗函数参数之一
bank=melbankm(M,N,fs,0,0.5,'m');

%归一化mel滤波器组系数
bank=full(bank);%full() convert sparse matrix to full matrix
bank=bank/max(bank(:));

%DCT系数，L*M
% L 阶指 MFCC 系数阶数，通常取 12-16
L = 12;
for k=1:L
    n=0:M-1;
    dctcoef(k,:)=cos((2*n+1)*k*pi/(2*M));
end

%归一化倒谱提升窗口
w=1+6*sin(pi*[1:L]./L);

%预加重滤波器
w=w/max(w);
xx=double(x);
xx=filter([1-0.9375],1,xx);

%语音信号分帧
xx=enframe(xx,N,80);

%计算每帧的MFCC参数
m=zeros(size(xx,1),L);%不初始化下面会出错
for i=1:size(xx,1)
    y=xx(i,:);
    s=y' .* hamming(N);
    t=abs(fft(s));
    t=t.^2;%计算能量
    c1=dctcoef*log(bank*t(1:129));%dctcoef为dct系数，bank归一化mel滤波器组系数
    c2=c1.*w';%w为归一化倒谱提升窗口
    m(i,:)=c2';
end

%一阶差分系数
dtm=zeros(size(m));
for i=3:size(m,1)-2
    dtm(i,:)=-2*m(i-2,:)-m(i-1,:)+m(i+1,:)+2*m(i+2,:);
end
dtm=dtm/3;

%合并mfcc参数和一阶差分mfcc参数
ccc=[m dtm];

%去除首位两帧，因为这两帧的一阶差分参数为0
ccc=ccc(3:size(m,1)-2,:);

subplot(2,1,1);
ccc_1=ccc(:,1);
plot(ccc_1);
title('MFCC');
ylabel('幅值');
title('一维数组及其幅值的关系');
[h,w]=size(ccc);
A=size(ccc);
subplot(2,1,2);
plot([1,w],A);
xlabel('维数');
ylabel('幅值');
title('维数于幅值的关系');
