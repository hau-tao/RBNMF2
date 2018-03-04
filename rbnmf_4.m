 function [W,SIGMA,H,S,LAMBDA] = rbnmf_gibb(X,K,T_burn_in,T_collection,...
    eta, r0, p0, c0,e0,f0,a,b,gamma0)


%load observation X (VxN)
%load Swimmer
%==============================Setup=====================================================
%========================================================================================
%matrix input size
[V,N] = size(X); 
%K = 16; %number of dictionaries K
%T_burn_in = 300; %number of iteration for model to get convergence
%T_collection = T_burn_in; %number of iteration for the model to infer the posterior
%hyper-parameters chosen
%p0=0.1;r0=0.1;eta=0.1;
%f0=1;e0=1;c0=1;
%b=0.01;a=0.01;
%gamma0=1;
r_k = 50/K*ones(K,1);p_j = 0.5*ones(1,N);c_j = 0.5*ones(1,N);pi_j=betarnd(a,b,1,N);
p_k=0.5*ones(1,K);
tempS=zeros(V,N);
tempH=zeros(K,N);
%==========================================================================
%Initialization W,H,S,r,c,p,q
W = rand(V,K); W = bsxfun(@rdivide,W,sum(W,1));%W(VxK)
H = 1/K*ones(K,N);%H(KxN)
S = gamrnd (a,b,V,N);%S(VxN)
r = rand;c=rand;p=rand;q=rand;
%==========================================================================
%Initialize SIGMA (KxK), and LAMBDA(N,N)
binary_SIGMA = randi([0 1], 1,K);
SIGMA= diag(binary_SIGMA); % SIGMA(KxK) with binary vector (1xK)
binary_LAMBDA = randi([0 1], 1,N);
LAMBDA = diag(binary_LAMBDA);%LAMBDA(N,N)with binary vector(1xN)
for t=1: T_burn_in + T_collection
%==========================Gibb Sampling===================================
%======================Sample A_vkj, B_vj==================================
%====Assign A_vj to K latent factors=======================================

[A_vk,A_kn]= mult_rand(X,W*SIGMA,H);
%[A_vk,A_kn]=mult_rand(X,horzcat(W*SIGMA,S(:,1)),vertcat(H,productS_LAMBDA));

%[B_vn,B_n]= mult_rand(X,sum(S(:,1),2),sum(LAMBDA(1,:),1));
productS_LAMBDA=zeros(1,N);
for j =1:N
    productS_LAMBDA(1,j)=S(1,j)*LAMBDA(j,j);
end
[B_vn,B_n]= mult_rand(X-round(W*SIGMA*H),S(:,1),diag(LAMBDA));

%====Sample W_(:,k)========================================================
 W = gamrnd(eta+A_vk,1);
 W= bsxfun(@rdivide,W,sum(W,1));

for k=1:K
    W(:,k)=gamrnd(eta+A_vk(:,k),1);
  

end
%=====Sample A_k and B_vk=================================================
[kk,~,counts_A] = find(A_kn);
    kj_a = zeros(size(counts_A));
    A_k = zeros(K,1);

    for k=1:K
       
        [A_k(k),kj_a(kk==k)] = CRT(counts_A(kk==k),r_k(k)); 
    end;
  [jj,~,counts_B] = find(B_n);
  n_b = zeros(size(counts_B));
  B_j = zeros(N,1);
 

  for j=1:N
       
        [B_j(j),n_b(jj==j)] = CRT(counts_B(jj==j),p_j(j));
  end
%=====Sample H_kj =========================================================

% for k =1:K
%     for j =1:N
%     H(k,j)=gamrnd(r_k(k)+A_kn(k,j),(SIGMA(k,k)+c_j(j))^(-1));
%     end
% end

for k=1:K
    for j=1:N
    H(k,j)=gamrnd(r_k(k)+A_kn(k,j),(SIGMA(k,k)+c_j(j)+eps).^(-1));
    end
end

%========Sample SIGMA and LAMBDA===========================================
% p=0.5;        %probability of success
% n=256;
% A=rand(n);
% A=(A<p)
%======================Sample LAMBDA=======================================
sum_col_B =sum(B_vn,1);
inzB=find(sum_col_B);
izB=find(~sum_col_B);
sumS = -sum(S,1);

for idx =1:numel(izB)
    j = izB(idx);
   % LAMBDA(j,j) = (LAMBDA(j,j)< pi_j(j)*exp(sumS(j))/(pi_j(j)*exp(sumS(j))+1-pi_j(j)));  
    LAMBDA(j,j) = binornd(1, pi_j(j)*exp(sumS(j))/(pi_j(j)*exp(sumS(j))+1-pi_j(j)+eps));
    
end
for idx =1:numel(inzB)
    j = inzB(idx);
    if(X(j,j)==0)
    LAMBDA(j,j) = 1;  
    else
    LAMBDA(j,j)=0;
    end 
end
%==========================Sample SIGMA====================================
sum_col_A=sum(A_vk,1);
inzA=find(sum_col_A);
izA=find(~sum_col_A);
sumH = -sum(H,2);
for idx =1:numel(izA)
    k = izA(idx);
    SIGMA(k,k) = binornd(1,p_k(k)*exp(sumH(k))/(p_k(k)*exp(sumH(k))+1-p_k(k)+eps));  
     
    
end

for idx =1:numel(inzA)
    k = inzA(idx);
    if(X(k,k)==0)
    SIGMA(k,k) = 1;  
    else
    SIGMA(k,k)=0;
    end 
end

% %==========================================================================
% %=======================Sample S_vj========================================
% for v =1:V
%     for j =1:N
%     S(v,j)=gamrnd(p_j(j)+B_vn(v,j),(LAMBDA(j,j)+q)^(-1));
%     end
% end

for v=1:V
    for j=1:N
    S(v,j)=gamrnd(p_j(j)+B_vn(v,j),(LAMBDA(j,j)+q+eps).^(-1));        
    end
    
end
% %=======================Sample r_k and p_j=================================
log_part_r = zeros(1,K);
for k=1:K
    for j =1:N
        log_part_r(k) = log_part_r(k) + log(c_j(j)./(SIGMA(k,k)+c_j(j)+eps));
    end
end
r_k= gamrnd(gamma0/V+A_k, (c0*ones(1,K)-log_part_r+eps)'.^(-1));
log_part_pj=zeros(N,1);
for j=1:N
   
        log_part_pj(j) = f0-V*log(q/(LAMBDA(j,j)+q+eps))+eps;
    
end
 p_j=gamrnd(B_j'+p0*ones(1,N), log_part_pj'.^-1);
% %=======================Sample cj,q,pk,pi_j=================================

c_j=gamrnd(e0*ones(1,N)+sum(r_k)*ones(1,N),(sum(H,1)+f0*ones(1,N)+eps).^(-1));
q=gamrnd(e0+V*sum(p_j),(sum(sum(S))+f0+eps)^(-1));
p_k=betarnd(diag(SIGMA)+a*ones(K,1),1*ones(K,1)-diag(SIGMA)+b*ones(K,1));
pi_j= betarnd(diag(LAMBDA)+a*ones(N,1),1*ones(N,1)-diag(LAMBDA)+b*ones(N,1));
end

