%% 基于蜣螂优化算法的三维无人机航迹优化
clc;
clear;
close all;

%% 创建地图
MapSizeX = 200; 
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
mesh(Map, 'FaceAlpha', 0.8);
hold on;
for i = 1:size(ThreatAreaRadius)
    [X, Y, Z] = cylinder(ThreatAreaRadius(i), 50);
    X = X + ThreatAreaPostion(i,1);
    Y = Y + ThreatAreaPostion(i,2);
    Z(2,:) = Z(2,:) + 50;
    mesh(X, Y, Z)
end
set(gca, 'FontName', 'Times New Roman');

%% 设置起始点
startPoint = [0,0,20];
endPoint = [200,200,20];
plot3(startPoint(1), startPoint(2), startPoint(3), 'ro', 'MarkerSize', 8);
text(startPoint(1), startPoint(2), startPoint(3), 'Start', 'Color', 'k', 'FontSize', 15, 'FontName', 'Times New Roman');
plot3(endPoint(1), endPoint(2), endPoint(3), 'r*', 'MarkerSize', 8);
text(endPoint(1), endPoint(2), endPoint(3), 'End', 'Color', 'k', 'FontSize', 15, 'FontName', 'Times New Roman');
title('Map Information', 'FontName', 'Times New Roman');
set(gca, 'FontName', 'Times New Roman');

%% 蜣螂优化参数设置
NodesNumber = 2;
dim = 2 * NodesNumber;
lb = [20.*ones(1, NodesNumber), 0.*ones(1, NodesNumber)];
ub = [180.*ones(1, NodesNumber), 50.*ones(1, NodesNumber)];
fobj = @(x) fun(x, NodesNumber, startPoint, endPoint, ThreatAreaPostion, ThreatAreaRadius);
SearchAgents_no = 30;
Max_iteration = 500;

% 依次运行各算法并绘制结果
%algorithms = {'MELGWO', 'GQPSO', 'IDBO', 'DTSMA', 'CPSOGSA', 'WOA', 'SCA', 'HHO', 'COA', 'CDO', 'OMA', 'SWO', 'ESC'};
algorithms = {'IDBO'};
for alg = algorithms
    fprintf('%s Optimization in progress:\n', alg{1});
    tic;
    [Best_score, Best_pos, curve] = feval(alg{1}, SearchAgents_no, Max_iteration, lb, ub, dim, fobj);
    runtime = toc;
    
    % 获取插值后的路径
    [X_seq, Y_seq, Z_seq, x_seq, y_seq, z_seq] = GetThePathLine(Best_pos, NodesNumber, startPoint, endPoint);
    
    % 绘制路径
    figure;
    mesh(Map, 'FaceAlpha', 0.8);
    hold on;
    for i = 1:size(ThreatAreaRadius)
        [X, Y, Z] = cylinder(ThreatAreaRadius(i), 50);
        X = X + ThreatAreaPostion(i,1);
        Y = Y + ThreatAreaPostion(i,2);
        Z(2,:) = Z(2,:) + 50;
        mesh(X, Y, Z);
    end
    plot3(X_seq, Y_seq, Z_seq, 'r.-', 'linewidth', 3);
    plot3(x_seq, y_seq, z_seq, 'bo');
    title(sprintf('The best path found by %s', alg{1}), 'FontName', 'Times New Roman');
    set(gca, 'FontName', 'Times New Roman');
    
    % 保存路径图像
    filename = sprintf('%s_path.jpg', alg{1});
    print('-djpeg', '-r600', filename);
    
    % 绘制迭代曲线
    figure;
    plot(curve, 'Color', 'b', 'linewidth', 2);
    grid on;
    legend(alg{1}, 'FontName', 'Times New Roman');
    title('Convergence Curve', 'FontName', 'Times New Roman');
    xlabel('Iterations', 'FontName', 'Times New Roman');
    ylabel('Best Path Length', 'FontName', 'Times New Roman');
    set(gca, 'FontName', 'Times New Roman');
    
    % 保存迭代曲线图像
    filename = sprintf('%s_convergence.jpg', alg{1});
    print('-djpeg', '-r600', filename);
    
    disp(['Best fitness value: ', num2str(Best_score)]);
    disp(['Runtime: ', num2str(runtime), ' seconds']);
end
