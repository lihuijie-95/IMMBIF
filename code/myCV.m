function [tcv fcv]=myCV(gnd,kfold,krand)%��������myCV

% Startified k-fold CV partition
% [fcv tcv]=myCV(gnd,kfold,krand)
% Inputs
%  gnd   - class labels  �����ǩ
%  kflod - k folds   k ��
%  krand - seed for random number  ���������
% 
% Outputs���tcv��fcv
%  tcv - training fold (kfold-1)
%  fcv - test fold (1)

c=length(unique(gnd));%c�ǲ�ͬ��ǩ�ĸ���
scv=cell(c,kfold);%����c�У�kfold�е�ϸ������
for i=1:c
    t=find(gnd==i);%�����ǩ����i��λ��
    rng(krand,'v5uniform');
    rp=randperm(length(t));%����[1,2,...length(t)]���������
    t=t(rp);%��rp���Ƹ�t
    a=fix(length(t)/kfold);%t�ĳ��Ⱥ�kfold���ȡ����0������
    for j=1:kfold-1
        scv{i,j}=t((j-1)*a+1:j*a);
    end
    scv{i,kfold}=t((kfold-1)*a+1:length(t));
end
fcv=cell(kfold,1);%fcv��tcvΪkfold��1�е�ϸ������
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