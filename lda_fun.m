function [W_lda]=lda_fun(X,m1,m2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%   [Wproj]=lda_fun(X,m1,m2)
% Performs principal component analysis on a data set X and
% returns the W of the projected points
%
% INPUT ARGUMENTS:
%   X:      lxN matrix whose columns are the data vectors.
%   m1:     Corresponds to the mean of the 1-st class 
%   m2:     Corresponds to the mean of the 2-nd class 
%
% OUTPUT ARGUMENTS:
%   W_lda:  W of the projected points with LDA algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[l,N]=size(X);
P_1=400/500;
P_2=100/500;

% Compute S1 and S2
Xw1_m1 = X(:,1:400) - m1;
Xw2_m2 = X(:,401:500) - m2;

S1 = 1/400*(Xw1_m1 * Xw1_m1');
S2 = 1/100*(Xw2_m2 * Xw2_m2');

% Computing Sw inverse
Sw = S1 + S2;
Sw_inv = inv(Sw);

%FInd and return Wproj
W_lda=Sw_inv*(m1-m2);