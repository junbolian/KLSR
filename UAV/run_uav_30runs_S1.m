clc; clear; close all;
clearvars;

restoredefaultpath; rehash toolboxcache;
addpath('/Users/zhengzikun/Desktop/UAV/alg'); % change to yours
addpath('/Users/zhengzikun/Desktop/UAV/uav'); % change to yours

% ===== UAV setup =====
NodesNumber = 4;
dim = 2*NodesNumber;

startPoint = [0,0,20];
endPoint   = [200,200,20];

ThreatAreaPostion = [ ...
    65,  78;
    86,  92;
    102, 118;
    120, 132;
    138, 152;
    156, 168;
    175, 188];

ThreatAreaRadius = [ ...
    11;
    12;
    9;
    10;
    13;
    10;
    9];

lb = [0*ones(1,NodesNumber), 0*ones(1,NodesNumber)];
ub = [200*ones(1,NodesNumber), 50*ones(1,NodesNumber)];

fobj = @(x) fun(x, NodesNumber, startPoint, endPoint, ThreatAreaPostion, ThreatAreaRadius);

% ===== experiment config =====
nRuns = 30;
N = 30;
MaxIt = 400;
Gmax  = 400;

methods = {'CSO','CSO_KLSR','JADE','JADE_KLSR','GA','GA_KLSR','PSO','PSO_KLSR'};
final_best = zeros(nRuns, numel(methods));
runtime_s  = zeros(nRuns, numel(methods));

%% store best position from each run/method
best_pos = nan(nRuns, numel(methods), dim);

for r = 1:nRuns
    fprintf('Run %d/%d\n', r, nRuns);

    rng(r);
    t = tic; [g, pos, ~] = CSO(N, MaxIt, lb, ub, dim, fobj);
    runtime_s(r,1)=toc(t); final_best(r,1)=g; best_pos(r,1,:)=pos(:);

    rng(r);
    t = tic; [g, pos, ~] = CSO_KLSR(N, MaxIt, lb, ub, dim, fobj);
    runtime_s(r,2)=toc(t); final_best(r,2)=g; best_pos(r,2,:)=pos(:);

    rng(r);
    t = tic; [g, pos, ~] = JADE(N, Gmax, lb, ub, dim, fobj);
    runtime_s(r,3)=toc(t); final_best(r,3)=g; best_pos(r,3,:)=pos(:);

    rng(r);
    t = tic; [g, pos, ~] = JADE_KLSR(N, Gmax, lb, ub, dim, fobj);
    runtime_s(r,4)=toc(t); final_best(r,4)=g; best_pos(r,4,:)=pos(:);

    rng(r);
    t = tic; [g, pos, ~] = GA(N, MaxIt, lb, ub, dim, fobj);
    runtime_s(r,5)=toc(t); final_best(r,5)=g; best_pos(r,5,:)=pos(:);

    rng(r);
    t = tic; [g, pos, ~] = GA_KLSR(N, MaxIt, lb, ub, dim, fobj);
    runtime_s(r,6)=toc(t); final_best(r,6)=g; best_pos(r,6,:)=pos(:);

    rng(r);
    t = tic; [g, pos, ~] = PSO(N, MaxIt, lb, ub, dim, fobj);
    runtime_s(r,7)=toc(t); final_best(r,7)=g; best_pos(r,7,:)=pos(:);

    rng(r);
    t = tic; [g, pos, ~] = PSO_KLSR(N, MaxIt, lb, ub, dim, fobj);
    runtime_s(r,8)=toc(t); final_best(r,8)=g; best_pos(r,8,:)=pos(:);
end

% summarize (mean/std)
mean_best = mean(final_best, 1);
std_best  = std(final_best, 0, 1);
mean_time = mean(runtime_s, 1);

Summary = table(methods', mean_best', std_best', mean_time', ...
    'VariableNames', {'Method','MeanBest','StdBest','MeanTime_s'});
disp(Summary);

% ===== save =====
save('/Users/zhengzikun/Desktop/UAV/uav_30runs_S1.mat', ...
     'final_best','runtime_s','methods','Summary', ...
     'NodesNumber','dim','lb','ub','N','MaxIt','Gmax', ...
     'startPoint','endPoint','ThreatAreaPostion','ThreatAreaRadius', ...
     'best_pos');  % save best positions too

writetable(Summary, '/Users/zhengzikun/Desktop/UAV/uav_30runs_S1_summary.csv');

%% ============================================================
% Pick run per method (STRICT seeds from this run)
% Use BEST run for each method (min final fitness)
% ============================================================

bestPos_all = nan(numel(methods), dim);
seed_used   = nan(numel(methods), 1);

for k = 1:numel(methods)
    fits = final_best(:,k);
    [~, order] = sort(fits, 'ascend');
    pick_idx = order(ceil(numel(order)/2));   % median-run index
    seed_used(k) = pick_idx;                  % seed was rng(r), so r=pick_idx
    bestPos_all(k,:) = squeeze(best_pos(pick_idx,k,:))';
end

disp(table(methods(:), seed_used, 'VariableNames', {'Method','BestRunSeedUsedForPlot'}));

%% ============================================================
% Build terrain once (reused by 3D + 2D plots)
% ============================================================

MapSizeX = 200; MapSizeY = 200;
Map = zeros(MapSizeX, MapSizeY);
for i = 1:MapSizeX
    for j = 1:MapSizeY
        Map(i,j) = MapValueFunction(i,j);
    end
end

%% ============================================================
% 3D overlay plot (same as before)
% ============================================================

C = lines(numel(methods));
figure('Color','w'); hold on;

mesh(Map, 'FaceAlpha', 0.35);

% threat cylinders
for i = 1:size(ThreatAreaRadius,1)
    [Xc, Yc, Zc] = cylinder(ThreatAreaRadius(i), 60);
    Xc = Xc + ThreatAreaPostion(i,1);
    Yc = Yc + ThreatAreaPostion(i,2);
    Zc(2,:) = Zc(2,:) + 50;
    mesh(Xc, Yc, Zc, 'FaceAlpha', 0.20);
end

% start/end
plot3(startPoint(1), startPoint(2), startPoint(3), 'kp', 'MarkerSize', 10, 'LineWidth', 2);
plot3(endPoint(1),   endPoint(2),   endPoint(3),   'k*', 'MarkerSize', 10, 'LineWidth', 2);

h = gobjects(numel(methods),1);
for k = 1:numel(methods)
    Best_pos = bestPos_all(k,:);
    [X_seq, Y_seq, Z_seq, ~, ~, ~] = GetThePathLine(Best_pos, NodesNumber, startPoint, endPoint);
    h(k) = plot3(X_seq, Y_seq, Z_seq, '-', 'Color', C(k,:), 'LineWidth', 3);
end

legend(h, strrep(methods,'_','\_'), 'Location','northeastoutside');
title('Scenario S1: overlay of 8 best paths', 'FontName','Times New Roman');
set(gca,'FontName','Times New Roman');
xlim([0 200]); ylim([0 200]); zlim([0 100]);
grid on; box on; view(45,25);

print('-djpeg','-r600','/Users/zhengzikun/Desktop/UAV/uav_S1_overlay_8paths.jpg');

%% ============================================================
% 2D top-view map + rings + overlay of 8 best paths
% ============================================================

figure('Color','w'); hold on;

% terrain background (top view)
imagesc(1:size(Map,2), 1:size(Map,1), Map);
axis xy; axis equal; axis tight;
colormap parula;

% threat rings (concentric circles)
theta = linspace(0, 2*pi, 400);
ring_scales = [1.0, 1.5, 2.0];

for i = 1:size(ThreatAreaRadius,1)
    cx = ThreatAreaPostion(i,1);
    cy = ThreatAreaPostion(i,2);
    R  = ThreatAreaRadius(i);

    for s = ring_scales
        plot(cx + (s*R)*cos(theta), cy + (s*R)*sin(theta), 'r-', 'LineWidth', 1);
    end
    plot(cx, cy, 'r.', 'MarkerSize', 12);
end

% start/end in 2D
plot(startPoint(1), startPoint(2), 'ks', 'MarkerSize', 8, 'LineWidth', 2);
plot(endPoint(1),   endPoint(2),   'k*', 'MarkerSize', 10, 'LineWidth', 2);

% 8 paths in 2D
h2 = gobjects(numel(methods),1);
for k = 1:numel(methods)
    Best_pos = bestPos_all(k,:);
    [X_seq, Y_seq, ~, ~, ~, ~] = GetThePathLine(Best_pos, NodesNumber, startPoint, endPoint);
    h2(k) = plot(X_seq, Y_seq, '-', 'Color', C(k,:), 'LineWidth', 2.5);
end

legend(h2, strrep(methods,'_','\_'), 'Location','eastoutside');
title('Scenario S1: overlay of 8 best paths (2D top view)', 'FontName','Times New Roman');
set(gca,'FontName','Times New Roman');
xlabel('x'); ylabel('y');

print('-djpeg','-r600','/Users/zhengzikun/Desktop/UAV/uav_S1_overlay_8paths_2D.jpg');
