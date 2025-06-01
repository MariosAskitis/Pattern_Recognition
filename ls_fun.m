function [W_ls]=ls_fun(X_ext,y)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FUNCTION
%   [W_ls]=ls_fun(X,m1,m2))
% Performs principal component analysis on a extended data set X and
% returns the W of a classified line based on the least squares.
%
% INPUT ARGUMENTS:
%   X:      Extended data vectors.
%   y:      Vector to indefy if a data point belong to 1-st class(y=1) or
%           it belong to 2-nd class(y=-1)
%
% OUTPUT ARGUMENTS:
%   W_lda:  W of the help of least squares
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

W_ls = (inv(X_ext' * X_ext)) * (X_ext' * y);