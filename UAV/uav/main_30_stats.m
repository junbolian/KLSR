clear;clc;close all

%% 基于蜣螂优化算法的三维无人机航迹优化
Runs = 30; % 独立运行次数
warning('off');

%% 创建地图
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
ThreatAreaPostion = [50,140];
ThreatAreaRadius = 30;

figure
mesh(Map);
hold on;
for i = 1:size(ThreatAreaRadius)
    [X,Y,Z] = cylinder(ThreatAreaRadius(i),50);
    X = X + ThreatAreaPostion(i,1);
    Y = Y + ThreatAreaPostion(i,2);
    Z(2,:) = Z(2,:) + 50;
    mesh(X,Y,Z)
end

%% 设置起始点
startPoint = [0,0,20];
endPoint = [200,200,20];
plot3(startPoint(1),startPoint(2),startPoint(3),'ro');
text(startPoint(1),startPoint(2),startPoint(3),'起点','Color','k','FontSize',15);
plot3(endPoint(1),endPoint(2),endPoint(3),'r*');
text(endPoint(1),endPoint(2),endPoint(3),'终点','Color','k','FontSize',15);
title('地图信息');

%% 蜣螂优化参数设置
NodesNumber = 2;
dim = 2 * NodesNumber;
lb = [20.*ones(1,NodesNumber),0.*ones(1,NodesNumber)];
ub = [180.*ones(1,NodesNumber),50.*ones(1,NodesNumber)];
fobj = @(x)fun(x,NodesNumber,startPoint,endPoint,ThreatAreaPostion,ThreatAreaRadius);
N = 30; % 种群数量
T = 500; % 设定最大迭代次数

%% 初始化结果存储
results = cell(1, 13); % 用于存储每个算法的结果
convergence = cell(1, 13); % 用于存储收敛曲线
algorithms = {'MELGWO','GQPSO','IDBO','DTSMA','CPSOGSA','WOA','SCA','HHO','COA','CDO','OMA','SWO','ESC'};

% 独立运行每个算法
for alg = 1:length(algorithms)
    results{alg} = zeros(Runs, 1);
    convergence{alg} = zeros(Runs, T+1);
    
    for run = 1:Runs
        initial_flag = 0; % 重置初始标志
        fprintf('%s is now tackling your problem, Run %d\n', algorithms{alg}, run);
        
        if strcmp(algorithms{alg}, 'ESC')
            BEF = inf; % 初始化一个大于400的值
            while BEF > 400 % 检查ESC的最优值是否大于400
                [BEF, ~, BestCost] = feval(algorithms{alg}, N, T, lb, ub, dim, fobj);
            end
        else
            [BEF, ~, BestCost] = feval(algorithms{alg}, N, T, lb, ub, dim, fobj);
        end
        
        results{alg}(run) = BEF;
        convergence{alg}(run, :) = BestCost;
    end
end

% 计算统计数据
outcomeMax = zeros(1, length(algorithms));
outcomeMin = zeros(1, length(algorithms));
outcomeMean = zeros(1, length(algorithms));
outcomeMedian = zeros(1, length(algorithms));
outcomeStd = zeros(1, length(algorithms));
outcomePvalue = zeros(1, length(algorithms));
for alg = 1:length(algorithms)
    outcomeMax(alg) = max(results{alg});
    outcomeMin(alg) = min(results{alg});
    outcomeMean(alg) = mean(results{alg});
    outcomeMedian(alg) = median(results{alg});
    outcomeStd(alg) = std(results{alg});
    
    % 秩和检验（与ESC算法比较）
    [pValue, ~] = ranksum(results{alg}, results{end});
    outcomePvalue(alg) = pValue;
end

% Friedman检验数据矩阵
alg_data_matrix = zeros(length(algorithms), Runs);
for i = 1:length(algorithms)
    alg_data_matrix(i,:) = results{i}';
end
alg_data_matrix = alg_data_matrix';

% 进行Friedman检验
if size(alg_data_matrix, 1) > 1
    [~, ~, stats] = friedman(alg_data_matrix, 1, 'off');
    outcomeFriedmanValue = stats.meanranks;
else
    error('不足以进行弗里德曼检验。');
end

% 保存到表格
folderPath = 'UAV path planning(case 2)';
if ~exist(folderPath, 'dir')
    mkdir(folderPath);
end
T = table(algorithms', outcomeMax', outcomeMin', outcomeMean', outcomeMedian', outcomeStd', outcomePvalue', outcomeFriedmanValue', ...
          'VariableNames', {'Algorithm', 'Max', 'Min', 'Mean', 'Median', 'Std', 'PValue', 'FriedmanValue'});
writetable(T, fullfile(folderPath, 'algorithm_performance.xlsx'));

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
colors = {color1, color2, color3, color4, color5, color6, color7, color8, color9, color10, color11, color12, color13};

% 绘制平均收敛曲线
figure;
hold on;
for alg = 1:length(algorithms)
    meanCurve = mean(convergence{alg}, 1); % 使用花括号访问元胞数组
    semilogy(meanCurve, 'Color', colors{alg}, 'LineWidth', 2);
end
legend({'MELGWO','GQPSO','IDBO','DTSMA','CPSOGSA','WOA','SCA','HHO','COA','CDO','OMA','SWO','ESC'}, 'Location', 'best');
CurveTitle = 'Convergence curve of UAV path planning(case 2)';
title(CurveTitle);
xlabel('Iteration');
ylabel('Average fitness value');
grid on;
saveas(gcf, fullfile(folderPath, 'convergence_curves.png'));
