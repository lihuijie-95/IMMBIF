function [tcv fcv]=myCV(gnd,kfold,krand)%创建函数myCV

% Startified k-fold CV partition
% [fcv tcv]=myCV(gnd,kfold,krand)
% Inputs
%  gnd   - class labels  分类标签
%  kflod - k folds   k 折
%  krand - seed for random number  随机数种子
% 
% Outputs输出tcv，fcv
%  tcv - training fold (kfold-1)
%  fcv - test fold (1)

c=length(unique(gnd));%c是不同标签的个数
scv=cell(c,kfold);%创建c行，kfold列的细胞数组
for i=1:c
    t=find(gnd==i);%分类标签等于i的位置
    rng(krand,'v5uniform');
    rp=randperm(length(t));%生成[1,2,...length(t)]的随机排列
    t=t(rp);%将rp复制给t
    a=fix(length(t)/kfold);%t的长度和kfold相除取靠近0的整数
    for j=1:kfold-1
        scv{i,j}=t((j-1)*a+1:j*a);
    end
    scv{i,kfold}=t((kfold-1)*a+1:length(t));
end
fcv=cell(kfold,1);%fcv、tcv为kfold行1列的细胞数组
tcv=cell(kfold,1);
for k=1:kfold,
    for i=1:c,
        fcv{k}=[fcv{k} scv{i,k}];
        t=[];
        for j=1:kfold,
            if k~=j
                t=[t scv{i,j}];
            end
        end
        tcv{k}=[tcv{k} t];
    end
end