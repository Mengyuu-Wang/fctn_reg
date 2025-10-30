%% plot
x = 60:20:240; %100:20:280; %140:20:320
 
set(groot, 'DefaultfigureRenderer', 'painters'); 
set(groot, 'DefaultfigurePosition', [100, 100, 800, 600]); 
resolution = 1000; 
set(gcf, 'GraphicsSmoothing', 'on'); 
% ==========================
a = 1;

plot(x, RMSEmat_fctn(a, :), '--+', 'Color', '#AB28FC', 'LineWidth', 2.3, 'MarkerSize', 13); 
hold on
plot(x, RMSEmat_tt(a, :), '-.s', 'Color', '#FF0A00', 'LineWidth', 2.3, 'MarkerSize', 13); 
%plot(x, RMSEmat_cp(a, :), '--d', 'Color', '#F08650',  'LineWidth', 2.3, 'MarkerSize', 13); 
plot(x, RMSEmat_tr(a, :), '-.*', 'Color', '#1C4EFC', 'LineWidth', 2.3, 'MarkerSize', 13); 
hold off

axis([60,240,0,1.5]);  %确定x轴与y轴框图大小
set(gca,'XTick',x);
set(gca,'FontSize',20);
set(gca,'FontName','Times New Roman');
%set(gca,'Yscale','log')  
set(gca,'YTick',[0 0.2 0.4 0.6 0.8 1 1.2 1.4]) 

%,'TTR-CP'
legend('FCTNRR-ALS-Fast','TTR-TT','TTR-TR', 'FontSize', 20,'Location', 'northeast','FontName', 'Times New Roman')   %右上角标注,'TW-KSRFT-ALS-Premix'
xlabel('number of samples (N)','FontSize', 25)  %x轴坐标描述
%ylabel('Time (seconds)','FontSize', 25) %y轴坐标描述
ylabel('RMSE','FontSize', 25) %y轴坐标描述
%title("Sketch sizes v.s. Time (seconds)",'FontSize', 25,'FontWeight', 'bold')
title("number of samples (N) v.s. RMSE",'FontSize', 25,'FontWeight', 'bold')
