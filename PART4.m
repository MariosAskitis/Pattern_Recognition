%%%%%%%%%%%%%%%%%%%%%%%%%%%  MEROS D  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
X_ext = [X; (ones(1,500))]';
Y =[(ones(1,400)) (-1)*(ones(1,100)) ]';
class1_data=w1';
[m1, S1]=Gaussian_ML_estimate(class1_data);
class2_data=w2';
[m2, S2]=Gaussian_ML_estimate(class2_data);

%%%%%%%%%%%%%%%%%%%%%%%%%%%  MEROS D1  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[W_ls]=ls_fun(X_ext,Y);

% Plot X
figure(1), hold on
p1 = plot(xRandom1, yRandom1, 'r.', 'MarkerSize', 10);
p2 = plot(xRandom2, yRandom2, 'g.', 'MarkerSize', 10);

% Plot the classifier (line) with least squares;
x = 0:0.01:12;
y = (-W_ls(1,:)*x - W_ls(3,:))/W_ls(2,:);
figure(1), p3= plot(x,y,'LineWidth',1.5);
figure(1), axis equal
set(gca, 'XAxisLocation', 'origin', 'YAxisLocation', 'origin')
legend([p1,p2,p3],'w1','w2', 'LS line');
xlim([0, 12]);
ylim([-2, 8]);
title('Least squares classifier of data W1, W2');

%compute classifier error
y_ls= [];

for i = 1:500
    if (W_ls(1,:)*X(1,i) + W_ls(2,:)*X(2,i) + W_ls(3,:)) > 0
        y_ls = ([y_ls, 1]);
    else 
        y_ls = ([y_ls, -1]);
    end
end

err_ls = (1-length(find(Y==y_ls'))/length(Y));
fprintf('Το σφάλμα ταξινόμησης του γραμμικού ταξινομητή (ευθεία) που ελαχιστοποιεί το κριτήριο ελαχίστων τετραγώνων ταξινόμησης είναι %.2f %%\n', (err_ls*100));

%compute least squares error
LSE=0;

for i = 1:500
    LSE = LSE + ((Y(1,:) - (X_ext(1,:)*W_ls))^2);
end
fprintf('Tο σφάλμα τετραγώνων του γραμμικού ταξινομητή (ευθεία) που ελαχιστοποιεί το κριτήριο ελαχίστων τετραγώνων ταξινόμησης είναι %f\n', LSE);


%%%%%%%%%%%%%%%%%%%%%%%%%%%  MEROS D2  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


rho=1; % Learning rate
w_ini=[0 1/2 0]';
[w_perce,iter,mis_clas]=perce(X_ext',Y',w_ini,rho)

% Plot X
figure(2), hold on
p1 = plot(xRandom1, yRandom1, 'r.', 'MarkerSize', 10);
p2 = plot(xRandom2, yRandom2, 'g.', 'MarkerSize', 10);

% Plot the classifier (line) with Perceptron;
x = 0:0.01:12;
y = (-w_perce(1,:)*x - w_perce(3,:))/w_perce(2,:);
figure(2), p3= plot(x,y, 'cy','LineWidth',1.5);
figure(2), axis equal
set(gca, 'XAxisLocation', 'origin', 'YAxisLocation', 'origin')
legend([p1,p2,p3],'w1','w2', 'Perceptron line');
xlim([0, 12]);
ylim([-2, 8]);
title('Perceptron classifier of data W1, W2');
movegui('east');

%compute classifier error
y_perce= [];

for i = 1:500
    if (w_perce(1,:)*X(1,i) + w_perce(2,:)*X(2,i) + w_perce(3,:)) > 0
        y_perce = ([y_perce, 1]);
    else 
        y_perce = ([y_perce, -1]);
    end
end

err_perce = (1-length(find(Y==y_perce'))/length(Y));
fprintf('Το σφάλμα ταξινόμησης του γραμμικού ταξινομητή (ευθεία) με βάση τον αλγόριθμο perceptron είναι %.2f %%\n', (err_perce*100));