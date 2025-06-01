%%%%%%%%%%%%%%%%%%%%%%%%%%%  MEROS C  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close('all');
clear;


%%%%%%%%%%%%%%%%%%%%%%% %CREATE AND PLOT W1, W2 %%%%%%%%%%%%%%%%%%%%%%%%%%%

% 1(a). To generate the data set X1 and a vector y1, whose i-th coordinate
% contains the class label of the i-th vector of X1, type
numPoints1 = 400;
widthx1 = 6;
widthy1 = 1;
x1 = 2;
y1 = 1;
xRandom1 = x1 + widthx1 * rand(1, numPoints1);
yRandom1 = y1 + widthy1 * rand(1, numPoints1);

numPoints2 = 100;
widthx2 = 2.5;
widthy2 = 3.5;
x2 = 5.5;
y2 = 2.5;
xRandom2 = x2 + widthx2 * rand(1, numPoints2);
yRandom2 = y2 + widthy2 * rand(1, numPoints2);

w1=[xRandom1(:) ,yRandom1(:)];
w2=[xRandom2(:) ,yRandom2(:)];
X=[w1; w2]';
y=[(ones(1,400,'uint16')) 2*(ones(1,100,'uint16')) ];
class1_data=w1';
[m1, S1]=Gaussian_ML_estimate(class1_data);
class2_data=w2';
[m2, S2]=Gaussian_ML_estimate(class2_data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%  MEROS C1  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% To compute the eigenvalues/eigenvectors and variance percentages required in this step type
m=2;
[eigenval,eigenvec,explained,Y,mean_vec]=pca_fun(X,m);

% The projections of the data points of X1 along the direction of the first
% principal component are contained in the first row of Y , returned by the
% function pca_fun above. 

% Plot X
figure(1), hold on
p1 = plot(xRandom1, yRandom1, 'r.', 'MarkerSize', 10);
p2 = plot(xRandom2, yRandom2, 'g.', 'MarkerSize', 10);

% Compute the projections of X
w=eigenvec(:,1);
t1=w'*X(:,1:400);
t2=w'*X(:,401:500);
X_proj1=[t1;t1].*((w/(w'*w))*ones(1,length(t1)));
X_proj2=[t2;t2].*((w/(w'*w))*ones(1,length(t2)));

% Plot the projections
figure(1), p3 = plot(X_proj1(1,:),X_proj1(2,:),'k.');
figure(1), p4 = plot(X_proj2(1,:),X_proj2(2,:),'ko');
figure(1), axis equal

set(gca, 'XAxisLocation', 'origin', 'YAxisLocation', 'origin')
legend([p1,p2,p3,p4],'w1','w2', 'projW1', 'projW2');
xlim([0, 12]);
ylim([-2, 8]);
title('PCA PROJECTION OF DATA W1, W2');

%%%%%%%%%%%%%%%%%%%%%%%%%%%  MEROS C2  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%compute m1, m2 for the X_proj1, X_proj2
class1_data=X_proj1;
[m_proj1, S_proj1]=Gaussian_ML_estimate(class1_data);
class2_data=X_proj2;
[m_proj2, S_proj2]=Gaussian_ML_estimate(class2_data);

%now we apply euclidean classifier with m_proj=[m_proj1 m_proj2]
m_proj=[m_proj1 m_proj2];
X_proj=[X_proj1 X_proj2];
z_euclidean_proj = euclidean_classifier(m_proj,X_proj);
err_euclidean_proj = (1-length(find(y==z_euclidean_proj))/length(y));
fprintf('Το λάθος της Ευκλείδιας ταξινόμησης για τις προβαλόμενες κλάσεις με PCA είναι %.2f %%\n', (err_euclidean_proj*100));

%%%%%%%%%%%%%%%%%%%%%%%%%%%  MEROS C3  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[W_lda]=lda_fun(X,m1,m2);

% Plot X
figure(2), hold on
p1 = plot(xRandom1, yRandom1, 'r.', 'MarkerSize', 10);
p2 = plot(xRandom2, yRandom2, 'g.', 'MarkerSize', 10);

% Compute the projections of X
t1_lda=W_lda'*X(:,1:400);
t2_lda=W_lda'*X(:,401:500);
X_lda1=[t1_lda;t1_lda].*((W_lda/(W_lda'*W_lda))*ones(1,length(t1_lda)));
X_lda2=[t2_lda;t2_lda].*((W_lda/(W_lda'*W_lda))*ones(1,length(t2_lda)));

% Plot the projections
figure(2), p5 = plot(X_lda1(1,:),X_lda1(2,:),'k.');
figure(2), p6 = plot(X_lda2(1,:),X_lda2(2,:),'ko');
figure(2), axis equal
set(gca, 'XAxisLocation', 'origin', 'YAxisLocation', 'origin')
legend([p1,p2,p5,p6],'w1','w2', 'projW1', 'projW2');
xlim([0, 12]);
ylim([-2, 8]);
title('LDA PROJECTION OF DATA W1, W2');
movegui('east');

%%%%%%%%%%%%%%%%%%%%%%%%%%%  MEROS C4  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%compute m1, m2 for the X_lda1, X_lda2
class1_data=X_lda1;
[m_lda1, S_lda1]=Gaussian_ML_estimate(class1_data);
class2_data=X_lda2;
[m_lda2, S_lda2]=Gaussian_ML_estimate(class2_data);

%now we apply euclidean classifier with m_lda=[m_lda1 m_lda2]
m_lda=[m_lda1 m_lda2];
X_lda=[X_lda1 X_lda2];
z_euclidean_lda = euclidean_classifier(m_lda,X_lda);
err_euclidean_lda = (1-length(find(y==z_euclidean_lda))/length(y));
fprintf('Το λάθος της Ευκλείδιας ταξινόμησης για τις προβαλόμενες κλάσεις με LDA είναι %.2f %%\n', (err_euclidean_lda*100));


