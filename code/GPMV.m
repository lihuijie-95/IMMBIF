function [modalcomp_comp, modalcomp_miss, modalmiss_comp, modalmiss_misscomp, modalmiss_miss, labelcomp, labelmiss] = GPMV(modalcomp,modalmiss,label,missrate)
% Low-Rank Representation based Incomplete Multi-Modal Brain Image Fusion for Epilepsy Classification
% Process data based on missing rate
% Author: LIHUIJIE
    nSmp = length(label);
    idx = randperm(nSmp);
    nMiss = floor(nSmp * missrate);
    nComp = nSmp - nMiss;
    
    modalcomp_comp = modalcomp(idx(1:nComp),:);
    modalcomp_miss = modalcomp(idx(nComp+1:end),:);
    
    modalmiss_comp = modalmiss(idx(1:nComp),:);
    modalmiss_misscomp = modalmiss(idx(nComp+1:end),:);
    
    modalmiss_miss = modalmiss_misscomp; modalmiss_miss(:,:) = 0;
    
    labelcomp = label(idx(1:nComp));
    labelmiss = label(idx(nComp+1:end));
    
end
