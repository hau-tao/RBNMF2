clc;clear all;close all;
%seed = 100; rand('state',seed); randn('state',seed);
%load  faces_1;
 %fea=imread('yaleB01_P00_Ambient.pgm');
 %img=imresize(img,[64 64]);
% % %   original = img;
% img(2:2:end,:,:) = 0;       %# Change every tenth row to black
% img(:,2:2:end,:) = 0;       %# Change every tenth column to black
 %figure(1),imshow(fea);                  %# Display the image
%img=imnoise(img,'salt & pepper',0.01);
% %  
% %  % figure(2),imshow(b)
% %  % b=im2double(b)
 %save('sbj.mat','fea');
% Nomalize each vector to unit
%===========================================
% fea=double(fea);
% [nSmp,nFea] = size(fea);
% for i = 1:nSmp
%      fea(i,:) = fea(i,:) ./ max(1e-12,norm(fea(i,:)));
% end
% %===========================================
% %Scale the features (pixel values) to [0,1]
% %===========================================
% maxValue = max(max(fea));
% fea = fea/maxValue;
% %===========================================
% % %  
% % % %seed = 100; rand('state',seed); randn('state',seed);
% % load sbj
%  X=fea;
% % X=double(X)
% %    
%    


 f=dir('*.pgm');
 f([f.isdir])=[];
 fil={f.name}; 
 %index=[4 29 30 31 32 33 ];
 index=[1:65];
 X=[];
  for k=1:numel(index)
      file=fil{index(k)};
      im=imread(file);
      im=imresize(im,[32 32]);
      im=im(:)';
      %im=reshape(im,size(im,1)*size(im,2),1);
      X= vertcat(X,im);
%       subplot(9,9,k);
%       imshow(im);
      
  end 
  X=double(X);
  X=X';
  X=sparse(X)
  %X= X./norm(X);
% function [W,SIGMA,H,S,LAMBDA] = rbnmf_4(X,K,T_burn_in,T_collection,...
%    eta, r0, p0, c0,e0,f0,a,b,gamma0)
K =16 ; %number of dictionaries K
base_num=K;
T_burn_in =100; %number of iteration for model to get convergence
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

% figure(111)
% for mm = 1:base_num
%     subplot(sqrt(base_num),sqrt(base_num),mm);imshow(reshape(X(:,mm),sqrt(M),sqrt(M)),[]);
%     
% end
% figure(444)
% for mm = 1:65
%     subplot(9,9,mm);imshow(reshape(X(:,mm),sqrt(M),sqrt(M)),[]);
%     
% end

figure(222)
for mm = 1:base_num
    subplot(sqrt(base_num),sqrt(base_num),mm);imshow(reshape(W(:,mm),sqrt(M),sqrt(M)),[]);
    
end


% 
% figure(333)
% for mm = 1:base_num
%     subplot((base_num),(base_num),mm);imagesc(reshape(W(:,mm),192,168),[0 max(max(W))]);colormap(gray);axis off;
% end

% figure(555)
% for ii = 1:size(W,2)
%     energ(ii) = norm(W(:,ii));
% end
% energ = energ./max(energ);
% latent = find(energ>0.3);
% stem(energ,'-');hold on;
% for ii = 1:length(latent)
% h = stem(latent(ii),energ(latent(ii)),'fill','-');set(h,'MarkerFaceColor','red');axis([0 base_num 0 1]);
% end