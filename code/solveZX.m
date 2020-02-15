function [X12, Z1, Z2] = solveZX(X11, X12, X21, X22, label_comp, label_miss, nIter, alpha, beta, epsilon_primal)
% Low-Rank Representation based Incomplete Multi-Modal Brain Image Fusion for Epilepsy Classification
% Incomplete modal   Complete part
% Incomplete modal   Incomplete part
% Complete modal   Corresponding  X11
% Complete modal   Corresponding  X12

% Author: LIHUIJIE

%% Initialization
rho = 0.1; 
miu = 1.01;
lamada = 0.1;
max_rho = 10^6;
[nComp,nFea] = size(X11);
[nMiss,~] = size(X12);
X12(:,:) = 0;
Z1 = zeros(nComp,nMiss);
Z2 = zeros(nComp,nMiss);
Z21 = zeros(nComp,nMiss);
E = zeros(nFea, nMiss);
y1 = zeros(nFea, nMiss);
y2 = zeros(nComp,nMiss);
I = eye(nComp);
Z2_pre = zeros(nComp,nMiss);

X11 = X11';
X12 = X12';
X21 = X21';
X22 = X22';
%% Iteration
for i = 1 : nIter
    
    % update Z21
    theta = beta / rho;
    temp_Z21 = Z2 + y2/rho ;
    [U, S, V] = svd( temp_Z21, 'econ');
    S = diag(S);
    svp = length(find(S > theta));
    if svp >= 1
        S = S(1:svp) - theta;
    else
        svp = 1;
        S = 0;    
    end
    Z21 = U(:,1:svp)*diag(S)*V(:,1:svp)';
    
    % update Z2
    Z2 = ( 2*alpha*I + rho*X21'*X21 + rho*I ) \ ( 2*alpha*Z1 + rho*X21'*X22 - rho*X21'*E + X21'*y1 + rho*Z21 - y2 );
    
    % update E
    xmaz = X22 - X21*Z2;
    temp = xmaz + y1/rho;
    E = solve_l1l2(temp,lamada/rho);
    
    % update Z1
    Z1 = ( alpha*I + X11'*X11 ) \ ( alpha*Z2 + X11'*X12 );
    
    % update X12
    X12 = X11* Z1;

    % update multiplies
    y1 = y1 + rho*( X22 - X21*Z2 - E);
    y2 = y2 + rho*(Z2 - Z21);
    
    % update rho
    rho = min(rho*miu, max_rho);
    
    % checking convergence
    primal1 = norm( X22 - X21*Z2, Inf );
    primal2 = norm( Z2 - Z21, Inf );
    if i > 2
        if ((primal1 < epsilon_primal) && (primal2 < epsilon_primal))
            break;
        end
    end
    
    Z2_pre = Z2;
end
[X12,PS1] = mapminmax(X12,0,1);
X12 = X12';

function [E] = solve_l1l2(W,lambda)
n = size(W,2);
E = W;
for i=1:n
    E(:,i) = solve_l2(W(:,i),lambda);
end

function [x] = solve_l2(w,lambda)
% min lambda |x|_2 + |x-w|_2^2
nw = norm(w);
if nw>lambda
    x = (nw-lambda)*w/nw;
else
    x = zeros(length(w),1);
end