%% 基于蜣螂优化算法的三维无人机航迹优化
%% 
clc;
clear;
close all;
%% 创建地图
%地图的大小200*200
MapSizeX = 200 ; 
MapSizeY = 200;
%% 地形地图创建,地图详细参数，请去MapValueFunction.m里面设置
x = 1:1:MapSizeX;
y = 1:1:MapSizeY;
for i = 1:MapSizeX
    for j = 1:MapSizeY
        Map(i,j) = MapValueFunction(i,j);
    end
end
global NodesNumber
global startPoint
global endPoint
global ThreatAreaPostion
global ThreatAreaRadius

%% 威胁区域绘制
%威胁区域中心坐标
ThreatAreaPostion = [50,140];
%威胁区域半径
ThreatAreaRadius = 30;
%将威胁区域叠加到图上
figure
mesh(Map);
hold on;
for i= 1:size(ThreatAreaRadius)
    [X,Y,Z] = cylinder(ThreatAreaRadius(i),50);
    X = X + ThreatAreaPostion(i,1);
    Y = Y + ThreatAreaPostion(i,2);
    Z(2,:) = Z(2,:) + 50;%威胁区域高度
    mesh(X,Y,Z)
end
%% 设置起始点
startPoint = [0,0,20];
endPoint = [200,200,20];
plot3(startPoint(1),startPoint(2),startPoint(3),'ro');
text(startPoint(1),startPoint(2),startPoint(3),'起点','Color','k','FontSize',15)
plot3(endPoint(1),endPoint(2),endPoint(3),'r*');
text(endPoint(1),endPoint(2),endPoint(3),'终点','Color','k','FontSize',15)
title('地图信息')
%% 蜣螂优化参数设置
NodesNumber = 2;%起点与终点之间节点的个数
dim = 2*NodesNumber; %维度，一组坐标点为[x,y,z]3个值，,其中X等间隔分布，所以总的数据个数为2*NodesNumber
lb = [20.*ones(1,NodesNumber),0.*ones(1,NodesNumber)];%x,y,z的下限[20,20,0]
ub = [180.*ones(1,NodesNumber),50.*ones(1,NodesNumber)];%x,y,z的上限[200,200,50]
fobj = @(x)fun(x,NodesNumber,startPoint,endPoint,ThreatAreaPostion,ThreatAreaRadius);%适应度函数
N=30; % 种群数量
Max_iteration=500; % 设定最大迭代次数

outcomeMean=[];  % 保存均值
outcomeStd=[];  % 保存标准差
outcomePvalue=[];  % 保存P值
outcomeFriedmanValue=[]; % 保存F值
outcomeFriedmanRank=[]; % 保存F排名
outcomeRunTime=[]; % 保存运行时间



%parfor i = 1:30 % 重复运行30次
for i = 1:30 % 重复运行30次

disp(['第',num2str(i),'次实验'])

disp('SHADE优化中: ')
tic
[sBest_SHADE(i,:), pBest_SHADE(i,:), conv_SHADE(i,:)] = SHADE(N, Max_iteration,lb,ub,dim,fobj);
SHADE_RunTime(i,:) = toc;

disp('LSHADE优化中: ')

tic
[sBest_LSHADE(i,:), pBest_LSHADE(i,:), conv_LSHADE(i,:)] = LSHADE(N, Max_iteration,lb,ub,dim,fobj);
LSHADE_RunTime(i,:) = toc;

disp('LSHADE_cnEpSin优化中: ')
tic
[sBest_LSHADE_cnEpSin(i,:), pBest_LSHADE_cnEpSin(i,:), conv_LSHADE_cnEpSin(i,:)] = LSHADE_cnEpSin(N, Max_iteration,lb,ub,dim,fobj);
LSHADE_cnEpSin_RunTime(i,:) = toc;

disp('SCA优化中: ')
tic
[sBest_SCA(i,:), pBest_SCA(i,:), conv_SCA(i,:)] = SCA(N,Max_iteration,lb,ub,dim,fobj);
SCA_RunTime(i,:) = toc;

disp('HHO优化中: ')
tic
[sBest_HHO(i,:), pBest_HHO(i,:), conv_HHO(i,:)] = HHO(N,Max_iteration,lb,ub,dim,fobj);
HHO_RunTime(i,:) = toc;


disp('WOA优化中: ')
tic
[sBest_WOA(i,:), pBest_WOA(i,:), conv_WOA(i,:)] = WOA(N,Max_iteration,lb,ub,dim,fobj);
WOA_RunTime(i,:) = toc;

disp('CDO优化中: ')
tic
[sBest_CDO(i,:), pBest_CDO(i,:), conv_CDO(i,:)] = CDO(N,Max_iteration,lb,ub,dim,fobj);
CDO_RunTime(i,:) = toc;

disp('SWO优化中: ')
tic
[sBest_SWO(i,:), pBest_SWO(i,:), conv_SWO(i,:)] = SWO(N,Max_iteration,lb,ub,dim,fobj);
SWO_RunTime(i,:) = toc;

disp('COA优化中: ')
tic
[sBest_COA(i,:), pBest_COA(i,:), conv_COA(i,:)] = COA(N,Max_iteration,lb,ub,dim,fobj);
COA_RunTime(i,:) = toc;

disp('AOA优化中: ')
tic
[sBest_AOA(i,:), pBest_AOA(i,:), conv_AOA(i,:)] = AOA(N,Max_iteration,lb,ub,dim,fobj);
AOA_RunTime(i,:) = toc;

disp('OMA优化中: ')
tic
[sBest_OMA(i,:), pBest_OMA(i,:), conv_OMA(i,:)] = OMA(N,Max_iteration,lb,ub,dim,fobj);
OMA_RunTime(i,:) = toc;

disp('NOA优化中: ')
tic
[sBest_NOA(i,:), pBest_NOA(i,:), conv_NOA(i,:)] = NOA(N,Max_iteration,lb,ub,dim,fobj);
NOA_RunTime(i,:) = toc;

disp('ESC优化中: ')
tic
[sBest_ESC(i,:), pBest_ESC(i,:), conv_ESC(i,:)] =ESC(N,Max_iteration,lb,ub,dim,fobj);
ESC_RunTime(i,:) = toc;
end


% 秩和检验p值
[pSHADE,~] = ranksum(sBest_SHADE,sBest_ESC); 
[pLSHADE,~] = ranksum(sBest_LSHADE,sBest_ESC);
[pLSHADE_cnEpSin,~] = ranksum(sBest_LSHADE_cnEpSin,sBest_ESC);
[pSCA,~] = ranksum(sBest_SCA,sBest_ESC);
[pHHO,~] = ranksum(sBest_HHO,sBest_ESC);
[pWOA,~] = ranksum(sBest_WOA,sBest_ESC);
[pCDO,~] = ranksum(sBest_CDO,sBest_ESC);
[pSWO,~] = ranksum(sBest_SWO,sBest_ESC);
[pCOA,~] = ranksum(sBest_COA,sBest_ESC);
[pAOA,~] = ranksum(sBest_AOA,sBest_ESC);
[pOMA,~] = ranksum(sBest_OMA,sBest_ESC);
[pNOA,~] = ranksum(sBest_NOA,sBest_ESC);
[pESC,~] = ranksum(sBest_ESC,sBest_ESC);

% Friedman值检验
data = [sBest_SHADE sBest_LSHADE sBest_LSHADE_cnEpSin sBest_SCA sBest_HHO sBest_WOA sBest_CDO sBest_SWO sBest_COA sBest_AOA sBest_OMA sBest_NOA sBest_ESC];
[~,~,Frk] = friedman(data,1,'off'); % 计算Friedman值

 % 对Friedman值进行排名
[~,I]=sort(Frk.meanranks);
[~,I]=sort(I);

outcomeMean1=[mean(sBest_SHADE) mean(sBest_LSHADE) mean(sBest_LSHADE_cnEpSin) mean(sBest_SCA) mean(sBest_HHO) mean(sBest_WOA) mean(sBest_CDO) mean(sBest_SWO) mean(sBest_COA) mean(sBest_AOA) mean(sBest_OMA) mean(sBest_NOA) mean(sBest_ESC)];
outcomeMean=vertcat(outcomeMean,outcomeMean1);  % 保存均值

outcomeStd1=[std(sBest_SHADE) std(sBest_LSHADE) std(sBest_LSHADE_cnEpSin) std(sBest_SCA) std(sBest_HHO) std(sBest_WOA) std(sBest_CDO) std(sBest_SWO) std(sBest_COA) std(sBest_AOA) std(sBest_OMA) std(sBest_NOA) std(sBest_ESC)];
outcomeStd=vertcat(outcomeStd,outcomeStd1);  % 保存标准差

outcomePvalue1=[pSHADE pLSHADE pLSHADE_cnEpSin pSCA pHHO pWOA pCDO pSWO pCOA pAOA pOMA pNOA pESC];
outcomePvalue=vertcat(outcomePvalue,outcomePvalue1);  % 保存P值

outcomeFriedmanValue=vertcat(outcomeFriedmanValue,Frk.meanranks); % 保存F值

outcomeFriedmanRank=vertcat(outcomeFriedmanRank,I); % 保存F排名

outcomeRunTime1=[mean(SHADE_RunTime) mean(LSHADE_RunTime) mean(LSHADE_cnEpSin_RunTime) mean(SCA_RunTime) mean(HHO_RunTime) mean(WOA_RunTime) mean(CDO_RunTime) mean(SWO_RunTime) mean(COA_RunTime) mean(AOA_RunTime) mean(OMA_RunTime) mean(NOA_RunTime) mean(ESC_RunTime)];
outcomeRunTime=vertcat(outcomeRunTime,outcomeRunTime1);  % 保存运行时间

folderPath = ([num2str(dim),'维运行结果']); % 文件夹路径

% 检查文件夹是否存在，如果不存在就创建
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end

% 合并文件名和文件夹路径
dataFilePath = fullfile(folderPath, ['路径规划','维运行结果.xlsx']);
writematrix(data, dataFilePath);

figure(1)
hold on;

% 绘制图形
% 定义颜色
%color10 = [1.0, 0.6, 0.6];   
%color9 = [0.68, 0.6, 0.76];  
%color8 = [0.36, 0.6, 0.84];  
%color7 = [1.0, 0.6, 0.36];    
%color6 = [0.68, 0.36, 0.76];  
%color5 = [0.36, 0.84, 0.84];  
%color4 = [1.0, 0.84, 0.36];    
%color3 = [0.52, 0.84, 1.0];   
%color2 = [0.36, 0.36, 0.76];  
%color1 = [1.0, 0.36, 0.36];   
% 用鲜艳的基本颜色
color1 = [1.0, 0.0, 0.0];  % 红色
color2 = [0.0, 0.0, 1.0];  % 蓝色
color3 = [0.0, 1.0, 0.0];  % 绿色
color4 = [1.0, 1.0, 0.0];  % 黄色
color5 = [1.0, 0.5, 0.0];  % 橙色
color6 = [0.5, 0.0, 0.5];  % 紫色
color7 = [0.0, 1.0, 1.0];  % 青色
color8 = [0.0, 0.5, 0.5];  % 暗青色
color9 = [0.5, 0.5, 0.5];  % 灰色
color10 = [1.0, 0.0, 1.0]; % 品红色
color11 = [0.04, 0.12, 0.42]; %blue
color12 =  [0.84, 0.34, 0.62]; %A blend of purple and pink 
color13 =  [0.46, 0.84, 0.06]; % A unique shade of green 
% 数据平均值计算
conv_SHADE = mean(conv_SHADE);
conv_LSHADE = mean(conv_LSHADE);
conv_LSHADE_cnEpSin = mean(conv_LSHADE_cnEpSin);
conv_SCA = mean(conv_SCA);
conv_HHO = mean(conv_HHO);
conv_WOA = mean(conv_WOA);
conv_CDO = mean(conv_CDO);
conv_SWO = mean(conv_SWO);
conv_COA = mean(conv_COA);
conv_AOA = mean(conv_AOA);
conv_OMA = mean(conv_OMA);
conv_NOA = mean(conv_NOA);
conv_ESC = mean(conv_ESC);

% 开始绘制图形
figure; hold on; % 保持绘图状态

% SHADE

semilogy(1:Max_iteration+1, conv_SHADE, '-*', 'Color', color1, 'Marker', '*', 'LineWidth', 1, 'MarkerSize', 2);

% LSHADE

semilogy(1:Max_iteration+1, conv_LSHADE, '-^', 'Color', color2, 'Marker', '^', 'LineWidth', 1, 'MarkerSize', 2);

% LSHADE_cnEpSin 

semilogy(1:Max_iteration+1, conv_LSHADE_cnEpSin, '-x', 'Color', color3, 'Marker', 'x', 'LineWidth', 1, 'MarkerSize', 2);

% SCA

semilogy(1:Max_iteration+1, conv_SCA, '-p', 'Color', color4, 'Marker', 'p', 'LineWidth', 1, 'MarkerSize', 2);

% HHO

semilogy(1:Max_iteration+1, conv_HHO, '-s', 'Color', color5, 'Marker', 's', 'LineWidth', 1, 'MarkerSize', 2);

% WOA

semilogy(1:Max_iteration+1, conv_WOA, '-o', 'Color', color6, 'Marker', 'o', 'LineWidth', 1, 'MarkerSize', 2);

% CDO
semilogy(1:Max_iteration+1, conv_CDO, '-d', 'Color', color7, 'Marker', 'd', 'LineWidth', 1, 'MarkerSize', 2);

% SWO

semilogy(1:Max_iteration+1, conv_SWO, '-+', 'Color', color8, 'Marker', '+', 'LineWidth', 1, 'MarkerSize', 2);

% COA

semilogy(1:Max_iteration+1, conv_COA, '-<', 'Color', color9, 'Marker', '<', 'LineWidth', 1, 'MarkerSize',2);

% AOA
semilogy(1:Max_iteration+1, conv_AOA, '-d', 'Color', color10, 'Marker', 'd', 'LineWidth', 1, 'MarkerSize', 2);

% OMA

semilogy(1:Max_iteration+1, conv_OMA, '-+', 'Color', color11, 'Marker', '+', 'LineWidth', 1, 'MarkerSize', 2);

% NOA

semilogy(1:Max_iteration+1, conv_NOA, '-<', 'Color', color12, 'Marker', '<', 'LineWidth', 1, 'MarkerSize',2);
% ESC

semilogy(1:Max_iteration+1, conv_ESC, '->', 'Color', color13, 'Marker', '>', 'LineWidth', 1, 'MarkerSize', 2);

% 调整图例顺序
legend('SHADE','LSHADE','LSHADE\_cnEpSin','SCA','HHO','WOA','CDO','SWO','COA','AOA','OMA','NOA','ESC');

title('Convergence curve')
xlabel('iterations');
ylabel('Average fitness value');
axis tight
grid off
box on
set(gca,'looseInset',[0 0 0 0],'FontSize',12);

% 定义文件名变量
filename = fullfile(folderPath, ['路径规划', '.jpg']);
% 保存为高质量 JPEG 文件
print('-djpeg', '-r600', filename);
close all

figure(2)
M=cat(1,sBest_SHADE', sBest_LSHADE', sBest_LSHADE_cnEpSin', sBest_SCA', sBest_HHO',  sBest_WOA' ,sBest_CDO',sBest_SWO',sBest_COA',sBest_AOA',sBest_OMA',sBest_NOA',sBest_ESC');
M1=M';
boxplot(M1,'Colors','b','Symbol','b+','Labels',{'SHADE','LSHADE','LSHADE_cnEpSin','SCA','HHO','WOA','CDO','SWO','COA','AOA','OMA','NOA','ESC'},'LabelOrientation','horizontal') ;
boxobj = findobj(gca,'Tag','Box');
title('Box plots')
set(gca,'looseInset',[0 0 0 0],'FontSize',12);
    



% 设置颜色
patch(get(boxobj(1),'XData'),get(boxobj(1),'YData'),color1,'FaceAlpha',0.5);
patch(get(boxobj(2),'XData'),get(boxobj(2),'YData'),color2,'FaceAlpha',0.5);
patch(get(boxobj(3),'XData'),get(boxobj(3),'YData'),color3,'FaceAlpha',0.5);
patch(get(boxobj(4),'XData'),get(boxobj(4),'YData'),color4,'FaceAlpha',0.5);
patch(get(boxobj(5),'XData'),get(boxobj(5),'YData'),color5,'FaceAlpha',0.5);
patch(get(boxobj(6),'XData'),get(boxobj(6),'YData'),color6,'FaceAlpha',0.5);
patch(get(boxobj(7),'XData'),get(boxobj(7),'YData'),color7,'FaceAlpha',0.5);
patch(get(boxobj(8),'XData'),get(boxobj(8),'YData'),color8,'FaceAlpha',0.5);
patch(get(boxobj(9),'XData'),get(boxobj(9),'YData'),color9,'FaceAlpha',0.5);   
patch(get(boxobj(10),'XData'),get(boxobj(10),'YData'),color10,'FaceAlpha',0.5);  
patch(get(boxobj(11),'XData'),get(boxobj(11),'YData'),color11,'FaceAlpha',0.5);
patch(get(boxobj(12),'XData'),get(boxobj(12),'YData'),color12,'FaceAlpha',0.5);   
patch(get(boxobj(13),'XData'),get(boxobj(13),'YData'),color13,'FaceAlpha',0.5);
% 定义文件名变量
filename = fullfile(folderPath, ['箱线图', '.jpg']);
% 保存为高质量 JPEG 文件
print('-djpeg', '-r600', filename);
close all



folderPath = ([num2str(dim),'维运行结果']); % 文件夹路径

% 检查文件夹是否存在，如果不存在就创建
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end


% 合并文件名和文件夹路径
meanFilePath = fullfile(folderPath, [num2str(dim), '维均值.xlsx']);
stdFilePath = fullfile(folderPath, [num2str(dim), '维标准差.xlsx']);
pValueFilePath = fullfile(folderPath, [num2str(dim), '维P值.xlsx']);
friedmanValueFilePath = fullfile(folderPath, [num2str(dim), '维F值.xlsx']);
friedmanRankFilePath = fullfile(folderPath, [num2str(dim), '维F排名.xlsx']);
runTimeFilePath = fullfile(folderPath, [num2str(dim), '维平均运行时间.xlsx']);

writematrix(outcomeMean, meanFilePath);
writematrix(outcomeStd, stdFilePath);
writematrix(outcomePvalue, pValueFilePath);
writematrix(outcomeFriedmanValue, friedmanValueFilePath);
writematrix(outcomeFriedmanRank, friedmanRankFilePath);
writematrix(outcomeRunTime, runTimeFilePath);

clear;clc;close all

outcomeMean=[];  % 保存均值
outcomeStd=[];  % 保存标准差
outcomePvalue=[];  % 保存P值
outcomeFriedmanValue=[]; % 保存F值
outcomeFriedmanRank=[]; % 保存F排名
outcomeRunTime=[]; % 保存运行时间
