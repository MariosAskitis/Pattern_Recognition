%%%%%%%%%%%%%%%%%%%%%%%%%%%  MEROS A  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close('all');
clear;

numPoints1 = 400;
widthx1 = 6;
widthy1 = 1;
x1 = 2;
y1 = 1;

figure(1)
rectangle('Position', [x1, y1, widthx1, widthy1],'LineWidth',1,'LineStyle','-');
hold on;
xRandom1 = x1 + widthx1 * rand(1, numPoints1);
yRandom1 = y1 + widthy1 * rand(1, numPoints1);
p1 = plot(xRandom1, yRandom1, 'r.', 'MarkerSize', 10);

numPoints2 = 100;
widthx2 = 2.5;
widthy2 = 3.5;
x2 = 5.5;
y2 = 2.5;
rectangle('Position', [x2, y2, widthx2, widthy2],'LineWidth',1,'LineStyle','-');
xRandom2 = x2 + widthx2 * rand(1, numPoints2);
yRandom2 = y2 + widthy2 * rand(1, numPoints2);
p2 = plot(xRandom2, yRandom2, 'g.', 'MarkerSize', 10);
set(gca, 'XAxisLocation', 'origin', 'YAxisLocation', 'origin')
legend([p1,p2],'w1','w2');
xlim([0, 10]);
ylim([-2, 8]);
title('Data points of two classes');
hold off;
w1=[xRandom1(:) ,yRandom1(:)];
w2=[xRandom2(:) ,yRandom2(:)];
X=[w1; w2];
y=[(ones(1,400,'uint16')) 2*(ones(1,100,'uint16')) ];