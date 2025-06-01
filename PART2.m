%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  MEROS B  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close('all');
clear;

%%%%%%%%%%%%%%%%%%%%%%% %CREATE AND PLOT W1, W2 %%%%%%%%%%%%%%%%%%%%%%%%%%%

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
X=[w1; w2];
y=[(ones(1,400,'uint16')) 2*(ones(1,100,'uint16')) ];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   B1.   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class1_data=w1';
[m1, S1]=Gaussian_ML_estimate(class1_data);

class2_data=w2';
[m2, S2]=Gaussian_ML_estimate(class2_data);

fprintf('Μέση τιμή για την κλάση 1');
m1
fprintf('Μητρώο συνδιασποράς για την κλάση 1');
S1
fprintf('Μέση τιμή για την κλάση 2');
m2
fprintf('Μητρώο συνδιασποράς για την κλάση 2');
S2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   B2.   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
S =(1/2)*(S1+S2);
m=[m1 m2];

%Employ the Euclidean distance classifier, using the ML estimates of the means, in order to classify the data vectors of X1
z_euclidean=euclidean_classifier(m,X');
err_euclidean = (1-length(find(y==z_euclidean))/length(y));
fprintf('Το λάθος της Ευκλείδιας ταξινόμησης είναι %.2f %%\n', (err_euclidean*100));

%we plot the missclasiffied points from euclidean_classifier with blue
%color in figure 2
figure(1)
rectangle('Position', [x1, y1, widthx1, widthy1],'LineWidth',1,'LineStyle','-');
hold on;
p1 = plot(xRandom1, yRandom1, 'r.', 'MarkerSize', 5);
rectangle('Position', [x2, y2, widthx2, widthy2],'LineWidth',1,'LineStyle','-');
p2 = plot(xRandom2, yRandom2, 'g.', 'MarkerSize', 5);
set(gca, 'XAxisLocation', 'origin', 'YAxisLocation', 'origin')
xlim([0, 10]);
ylim([-2, 8]);
title('Missclasiffied points from euclidean classifier');
missclasd_eukl = [];

for i = 1:500
    if y(i) ~= z_euclidean(i)
        missclasd_eukl = ([missclasd_eukl, i]);
    end
end


for j = 1:size(missclasd_eukl,2)
    if missclasd_eukl(j) <= 400
        w1_x_miss = w1(missclasd_eukl(j),1);
        w1_y_miss = w1(missclasd_eukl(j),2);
        p3 = plot(w1_x_miss,w1_y_miss,'b.', 'MarkerSize', 15);
        hold on;
    else
        w2_x_miss = w2((missclasd_eukl(j)-400),1);
        w2_y_miss = w2((missclasd_eukl(j)-400),2);
        p3 = plot(w2_x_miss,w2_y_miss,'b.', 'MarkerSize', 15);
        hold on;
    end
end
legend([p1,p2, p3],'w1','w2', 'euclidean miss');
hold off;
movegui('east');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   B3.   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Similarly, for the Mahalanobis distance classifier, we have
z_mahalanobis=mahalanobis_classifier(m,S,X');
err_mahalanobis = (1-length(find(y==z_mahalanobis))/length(y));
fprintf('Το λάθος της Mahalanobis ταξινόμησης είναι %.2f %%\n', (err_mahalanobis*100));

%we plot the missclasiffied points from Mahalanobis classifier with magenta
%color in figure 3
figure(2)
rectangle('Position', [x1, y1, widthx1, widthy1],'LineWidth',1,'LineStyle','-');
hold on;
p1 = plot(xRandom1, yRandom1, 'r.', 'MarkerSize', 5);
rectangle('Position', [x2, y2, widthx2, widthy2],'LineWidth',1,'LineStyle','-');
p2 = plot(xRandom2, yRandom2, 'g.', 'MarkerSize', 5);
set(gca, 'XAxisLocation', 'origin', 'YAxisLocation', 'origin')
xlim([0, 10]);
ylim([-2, 8]);
title('Missclasiffied points from Mahalanobis classifier');
movegui('west');

missclasd_mahal = [];

for i = 1:500
    if y(i) ~= z_mahalanobis(i)
        missclasd_mahal = ([missclasd_mahal, i]);
    end
end


for j = 1:size(missclasd_mahal,2)
    if missclasd_mahal(j) <= 400
        w1_x_miss = w1(missclasd_mahal(j),1);
        w1_y_miss = w1(missclasd_mahal(j),2);
        p3 = plot(w1_x_miss,w1_y_miss,'m.', 'MarkerSize', 15);
        hold on;
    else
        w2_x_miss = w2((missclasd_mahal(j)-400),1);
        w2_y_miss = w2((missclasd_mahal(j)-400),2);
        p3 = plot(w2_x_miss,w2_y_miss,'m.', 'MarkerSize', 15);
        hold on;
    end
end
legend([p1,p2, p3],'w1','w2', 'Mahalanobis miss');
hold off;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   B4.   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P=[400/500 100/500]'; 
S(:,:,1)=S1;S(:,:,2)=S2;

%For the Bayesian classifier, use function bayes classifier and provide as input the matrices m, S, P, which were used for the dataset generation.
z_bayesian=bayes_classifier(m,S,P,X');
err_bayesian = (1-length(find(y==z_bayesian))/length(y));
fprintf('Το λάθος της Bayesian ταξινόμησης είναι %.2f %%\n', (err_bayesian*100));
