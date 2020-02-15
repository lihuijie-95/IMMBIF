function [res] = cal_res(result_test, test_label)
% Low-Rank Representation based Incomplete Multi-Modal Brain Image Fusion for Epilepsy Classification
% Calculate ACC, SEN, SPE, PPV, NPV, F1, BAC
% Author: LIHUIJIE
    acc=0;
    tp=0;
    tn=0;
    fn=0;
    fp=0;
    test_num = length(test_label);
    for ii = 1:test_num
        if test_label(ii)== result_test(ii) 
            acc=acc+1;
        end
        if test_label(ii)== 1 && result_test(ii)== 1
            tp=tp+1;
        end
        if test_label(ii)== 1 && result_test(ii)== 0
            fn=fn+1;
        end
        if test_label(ii)== 0 && result_test(ii)== 1
            fp=fp+1;
        end
        if test_label(ii)== 0 && result_test(ii)== 0
            tn=tn+1;
        end
    end
    
    sen=tp/(tp+fn);  
    spe=tn/(fp+tn);  
    pp=tp/(tp+fp);
    np=tn/(tn+fn);
    acc1=acc/test_num;
    f1 = (2 * sen * pp) / (sen + pp);
    bac = (sen+spe)/2;
    
    res(1)=acc1;
    res(2)=sen;
    res(3)=spe;
    res(4)=pp;
    res(5)=np;
    res(6)=f1;
    res(7)=bac; 
end