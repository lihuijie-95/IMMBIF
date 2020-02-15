
clear;
clc;
C = 1;
module_num = 2;
kernel_num = 4;
missrate = 0;
k_fold = 10;
lnnum  = 10;
data_normalization = 1;

load data.mat;
feas{1} = modal1; % comp
feas{2} = modal2; % miss
fea = feas;
labels=label;

num_samples = size(modal1,1);
label=labels;
label(find(labels==0))=2;

for ln = 1 : lnnum
    i = 1;
    feas = fea;
    [tcv, fcv]=myCV(label',k_fold,ln);
    fea1 = feas{1};
    fea2 = feas{2};
    for index_beta = 10 % modal2
        res_testlabel = [];
        testlabel_k = [];
        for ki = 1:k_fold
            trlabel=tcv{ki};
            telabel=fcv{ki};
            train_labels = labels(trlabel,:);
            test_labels = labels(telabel,:);
            train_data1 = fea1(trlabel,:);
            test_data1 = fea1(telabel,:);
            train_data2 = fea2(trlabel,:);
            test_data2 = fea2(telabel,:);
            
            [modalcomp_comp, modalcomp_miss, modalmiss_comp, modalmiss_misscomp, modalmiss_miss, labelcomp, labelmiss] = GPMV(train_data1,train_data2,train_labels,missrate);
            train_features{1} = modalcomp_comp;
            train_features{2} = modalmiss_comp;
            test_features{1} = test_data1;
            test_features{2} = test_data2;
            train_labelgai = labelcomp;
            train_kernel_matrix = ones(size(train_labelgai,1),size(train_labelgai,1),module_num);
            test_kernel_matrix = ones(size(test_labels,1),size(train_labelgai,1),module_num);
            for j = 1:module_num
                train_data = train_features{j};
                test_data = test_features{j};
                
                train_km = generate_kernel(train_data,'jcb');
                test_km = generate_kernel(test_data,'jcb',1,train_data);
                if data_normalization
                    train_km = train_km/mean(mean(train_km));
                    test_km = test_km/mean(mean(test_km));
                end
                train_kernel_matrix(:,:,j) = train_km;
                test_kernel_matrix(:,:,j) = test_km;
            end
            beta = [index_beta/10,1-index_beta/10];
            train_data = combine_multi_kernel(train_kernel_matrix,beta);
            test_data = combine_multi_kernel(test_kernel_matrix,beta);
            train_data = [(1:size(train_data,1))',train_data];
            test_data = [(1:size(test_data,1))',test_data];
            
            [Y_new]=svm_once(train_data,test_data,train_labelgai,...
                test_labels,kernel_num,C);
            res_testlabel = [res_testlabel; Y_new];
            testlabel_k = [testlabel_k; test_labels];
            res(ki,:) = cal_res(Y_new,test_labels);
        end
        res_all(ln,i,:,:) = res;
        res_testlabel_all(ln,i,:) = res_testlabel;
        testlabel_all(ln,i,:) = testlabel_k;
        [auc,~,~] = plot_roc(res_testlabel,testlabel_k);
        ave_all(ln,i,:) = [mean(res,1) auc std(res,1,1) auc beta missrate k_fold ln];
        
        i = i + 1;
        
    end
    if i == 2
        best_ave_all(ln,:) = squeeze(ave_all(ln,:,:));
        bestROClabel_all(ln,:) = res_testlabel;
        besttlabel_all(ln,:) = testlabel_k;
        bestres_all(ln,:,:) = res;
    else
        aveln = squeeze(ave_all(ln,:,:));
        test_ROC = squeeze(res_testlabel_all(ln,:,:));
        test_tlabel = squeeze(testlabel_all(ln,:,:));
        resln = squeeze(res_all(ln,:,:,:));
        accln = aveln(:,1);
        aa = max(accln);
        bb = find(accln >= aa);
        if length(bb) > 1
            cc = aveln(bb,:);
            ccROC = test_ROC(bb,:);
            cctlabel = test_tlabel(bb,:);
            ccres = resln(bb,:,:);
            locauc = 8;
            dd = cc(:, locauc);
            ee = max(dd);
            ff = find(dd >= ee);
            if length(ff) > 1
                gg = ff(1);
                best = cc(gg,:);
                bestROClabel = ccROC(gg,:);
                besttlabel = cctlabel(gg,:);
                besrres = ccres(gg,:,:);
            else
                best = cc(ff,:);
                bestROClabel = ccROC(ff,:);
                besttlabel = cctlabel(ff,:);
                besrres = ccres(ff,:,:);
            end
        else
            best = aveln(bb,:);
            bestROClabel = test_ROC(bb,:);
            besttlabel = test_tlabel(bb,:);
            besrres = resln(bb,:,:);
        end
        best_ave_all(ln,:) = best;
        bestROClabel_all(ln,:) = bestROClabel;
        besttlabel_all(ln,:) = besttlabel;
        bestres_all(ln,:,:) = besrres;
    end
end
best_preave = best_ave_all(:,1:8);
best_ave = [mean(best_preave,1) std(best_preave,1,1)];
SingelModal2_ACC = best_ave(1)
SingelModal2_AUC = best_ave(8)
% save('resSingleModal2.mat','best_ave','best_ave_all','bestROClabel_all','besttlabel_all','bestres_all');
