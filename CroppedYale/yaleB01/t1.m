clc;clear all;close all;
[I, map] = imread('yaleB01_P00A+000E-35.pgm');

im=imresize(I,[32 32]);
im=im(:)';
X= imbinarize(im);
X=X';

%X = X / 255;
% v = var(X(:));
% J = imnoise(X, 'gaussian', 0, v / 10);
% figure(321);
imshow(reshape(X,32,32))
K =2 ; %number of dictionaries K
base_num=1;
T_burn_in =500; %number of iteration for model to get convergence
T_collection = T_burn_in; %number of iteration for the model to infer the posterior
%hyper-parameters chosen
p0=0.1;r0=0.1;eta=0.1;
f0=1;e0=1;c0=1;
b=0.1;a=0.1;
gamma0=1;
M = size(X,1);                                           % initialization
tic
[W,SIGMA,H,S,LAMBDA] = rbnmf_4(X,K,T_burn_in,T_collection,...
    eta, r0, p0, c0,e0,f0,a,b,gamma0);
toc
X_rec=W*H;
figure(222)
for mm = 1:base_num
    subplot(sqrt(base_num),sqrt(base_num),mm);imshow(reshape(X_rec(:,mm),sqrt(M),sqrt(M)),[]);
    
end
