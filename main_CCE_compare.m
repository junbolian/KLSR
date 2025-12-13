function main_CCE_compare
% ============================================================
%  Baseline (PSO/DE) vs CCE (Competitive Cluster Elimination)
%  - Grid-search CCE over (tau, rho); each combo runs 20 trials
%  - Pick best (tau,rho) per function by median best fitness
%  - Beautiful 3D bar & surface (tau × rho), convergence, boxplot
%  - Save all to MAT + PNGs
%
%  Dependencies (same folder):
%    CCE.m, PSO.m, DE.m, Get_Functions_cec2017.m, initialization.m
%  Author: Junbo Jacob Lian
% ============================================================

clc; clear; close all; warning off

%% ------------- Choose algorithm family -------------
algoMode  = 'PSO';     % 'PSO' or 'DE'

%% ------------- Experiment setup -------------
nPop      = 100;
dim       = 100;                           
MaxIt     = (10000*dim)/nPop;     % 确保总评估为 10000*dim
run_times = 20;
FuncList  = 1:30;                  % CEC-2017 F1..F30
skipFuncs = [2];                   % ★ 跳过 F02

% CCE grid
tau_grid  = [2 3 5];               % 刷新间隔
rho_grid  = [0.3 0.5 0.7 0.9];     % 最差簇重置比例
K_fixed   = [];                    % [] => auto: min(8, ceil(sqrt(N)))
steps_km  = 5;                     % k-means 内循环步数
pick_rule = 'min';                 % 'min' 或 'median'

% Aesthetics
set(0,'DefaultAxesFontName','Helvetica','DefaultTextFontName','Helvetica');
convCols = [0.15 0.15 0.15; 0.00 0.45 0.74];  % Baseline 深灰, CCE 蓝
try, cmap = turbo(256); catch, cmap = parula(256); end

% Reproducibility
master_seed = 20250810;            % 全局主种子

% 输出 tag
Tag = sprintf('CCE_%s_CEC2017_D%d', upper(algoMode), dim);

% 结果容器
All = struct();

for F = FuncList
    if ismember(F, skipFuncs)
        fprintf('\n========== %s  F%02d  (dim=%d) ==========\n', upper(algoMode), F, dim);
        fprintf('>> Skipping F%02d as requested.\n', F);
        continue;
    end

    % 获取函数
    [lb, ub, dim_true, fobj] = Get_Functions_cec2017(F, dim);
    fprintf('\n========== %s  F%02d  (dim=%d) ==========\n', upper(algoMode), F, dim_true);

    % 为该函数创建固定的 run 种子，保证 baseline 与所有网格组合可复现且公平
    rng(master_seed + F);
    base_seeds = randi(2^31-1, run_times, 1);

    %% -------- Baseline runs (20 trials) --------
    baseScores   = zeros(run_times,1);
    baseCurveSum = zeros(1, MaxIt);
    baseTimes    = zeros(run_times,1);

    for r = 1:run_times
        rng(base_seeds(r));                 % ★ 固定每次 run 的种子
        t0 = tic;
        [bestScore, curve] = run_baseline(algoMode, nPop, MaxIt, lb, ub, dim_true, fobj);
        baseTimes(r)  = toc(t0);
        baseScores(r) = bestScore;
        baseCurveSum  = baseCurveSum + curve(:)';    % accumulate
    end
    baseCurveAvg = baseCurveSum / run_times;
    baseMed      = median(baseScores);
    baseMeanTime = mean(baseTimes);
    fprintf('Baseline:  median=%.3e  meanTime=%.2fs/run\n', baseMed, baseMeanTime);

    %% -------- Grid search over (tau, rho) --------
    A = numel(tau_grid); B = numel(rho_grid);
    medGrid   = nan(A,B);               % rows: tau, cols: rho
    timeGrid  = nan(A,B);
    Scores    = cell(A,B);              % 缓存每个组合 20 次的 bestScore
    CurveSum  = cell(A,B);              % 缓存每个组合的收敛曲线和（用于平均）

    for ia = 1:A
        for ib = 1:B
            opts = struct('tau',tau_grid(ia), 'rho',rho_grid(ib), ...
                          'K',K_fixed, 'steps',steps_km, 'pick',pick_rule, ...
                          'keep_one',true, 'avoid_best',true);

            bestVec  = zeros(run_times,1);
            curveSum = zeros(1, MaxIt);
            t1 = tic;
            for r = 1:run_times
                rng(base_seeds(r));     % ★ 所有组合使用相同的种子序列
                if strcmpi(algoMode,'PSO')
                    [bestScore, curve] = run_pso_cce(nPop, MaxIt, lb, ub, dim_true, fobj, opts);
                else
                    [bestScore, curve] = run_de_cce (nPop, MaxIt, lb, ub, dim_true, fobj, opts);
                end
                bestVec(r)  = bestScore;
                curveSum    = curveSum + curve(:)';
            end
            medGrid(ia,ib)  = median(bestVec);
            timeGrid(ia,ib) = toc(t1)/run_times;
            Scores{ia,ib}   = bestVec;         % ★ 缓存
            CurveSum{ia,ib} = curveSum;        % ★ 缓存

            fprintf('Grid  tau=%d  rho=%.2f  median=%.3e  time=%.2fs/run\n', ...
                tau_grid(ia), rho_grid(ib), medGrid(ia,ib), timeGrid(ia,ib));
        end
    end

    % 选择网格最优
    [bestMed, idx] = min(medGrid(:));
    [ia_best, ib_best] = ind2sub(size(medGrid), idx);
    best_tau = tau_grid(ia_best);
    best_rho = rho_grid(ib_best);
    fprintf('>> Grid-best CCE on F%02d: tau=%d, rho=%.2f, median=%.3e\n', ...
            F, best_tau, best_rho, bestMed);

    %% -------- 直接复用网格阶段最佳组合的缓存（不二次重跑） --------
    cceScores   = Scores{ia_best, ib_best};
    cceCurveAvg = CurveSum{ia_best, ib_best} / run_times;
    cceMed      = medGrid(ia_best, ib_best);
    cceMeanTime = timeGrid(ia_best, ib_best);
    fprintf('CCE(best, cached): median=%.3e  meanTime=%.2fs/run\n', cceMed, cceMeanTime);

    %% -------- Store --------
    All(F).F              = F;
    All(F).algo           = upper(algoMode);
    All(F).baseline.med   = baseMed;
    All(F).baseline.scores= baseScores;
    All(F).baseline.curve = baseCurveAvg;
    All(F).baseline.time  = baseTimes;
    All(F).grid.tau_grid  = tau_grid;
    All(F).grid.rho_grid  = rho_grid;
    All(F).grid.medGrid   = medGrid;
    All(F).grid.timeGrid  = timeGrid;
    All(F).best.tau       = best_tau;
    All(F).best.rho       = best_rho;
    All(F).best.med       = bestMed;
    All(F).cce.med        = cceMed;
    All(F).cce.scores     = cceScores;
    All(F).cce.curve      = cceCurveAvg;
    All(F).cce.time       = cceMeanTime;

    %% -------- Plots: Convergence & Boxplot --------
    fig = figure('Color','w'); hold on
    semilogy(baseCurveAvg, 'LineWidth', 2.0, 'Color', convCols(1,:));
    semilogy(cceCurveAvg,  'LineWidth', 2.0, 'Color', convCols(2,:));
    grid on; box on
    xlabel('Iteration'); ylabel('Best Fitness (log)');
    title(sprintf('%s  F%02d (Dim=%d)', upper(algoMode), F, dim_true), 'FontWeight','normal');
    legend({'Baseline', sprintf('CCE(\\tau=%d, \\rho=%.2f)', best_tau, best_rho)}, 'Location','northeast');
    saveas(fig, sprintf('%s-F%02d-Conv.png', Tag, F));
    close(fig);

    fig = figure('Color','w','Position',[420 240 360 220]);
    boxplot([baseScores, cceScores], 'Labels', {'Baseline','CCE(best)'});
    set(gca,'YScale','log'); grid on; box on
    ylabel('Best Fitness (log)');
    title(sprintf('F%02d (Dim=%d)', F, dim_true), 'FontWeight','normal');
    saveas(gcf, sprintf('%s-F%02d-Box.png', Tag, F));
    close(gcf);

    %% -------- 3D: tau × rho → log10(median best) --------
    % medGrid is A×B (rows=tau, cols=rho). Build matrix with rows=rho (incl. 0), cols=tau.
    rhoAxis = [0, rho_grid];      % baseline at rho=0
    tauAxis = tau_grid;

    medMat = nan(numel(rhoAxis), numel(tauAxis));  % rows: rho, cols: tau
    medMat(1, :)     = baseMed;                    % baseline 复制到每个 tau
    medMat(2:end, :) = medGrid.';                  % 转置以行=rho, 列=tau

    Z = log10(max(medMat, eps));

    % ticks as powers of 10
    zmin = floor(min(Z(:)));
    zmax = ceil(max(Z(:)));
    zticks   = linspace(zmin, zmax, max(3, min(7, zmax - zmin + 1)));
    zticklbl = arrayfun(@(v) sprintf('10^{%.0f}', v), zticks, 'uni', 0);

    fig = figure('Color','w','Position',[180 90 980 430]);

    % ---- 3D Bar (rows=rho, cols=tau)
    subplot(1,2,1);
    bh = bar3(Z, 'detached');
    for k = 1:numel(bh), set(bh(k),'EdgeColor',[0.35 0.35 0.35]); end
    colormap(cmap); caxis([zmin zmax]); colorbar;
    grid on; box on
    set(gca,'ZTick',zticks,'ZTickLabel',zticklbl);
    set(gca,'XTick',1:numel(tauAxis), 'XTickLabel', arrayfun(@(x)sprintf('%d',x),tauAxis,'uni',0));
    set(gca,'YTick',1:numel(rhoAxis), 'YTickLabel', arrayfun(@(y)sprintf('%.2f',y),rhoAxis,'uni',0));
    xlabel('\tau'); ylabel('\rho'); zlabel('Median Best (log)');
    title(sprintf('F%d 3D Bar (Baseline at \\rho=0)', F), 'FontWeight','normal');
    view([-40 28]); axis tight;

    % ---- 3D Surface (match sizes: [length(rhoAxis) × length(tauAxis)])
    subplot(1,2,2);
    [TAU, RHO] = meshgrid(tauAxis, rhoAxis);
    surf(TAU, RHO, Z, 'EdgeAlpha',0.2, 'FaceAlpha',0.95);
    colormap(cmap); caxis([zmin zmax]); colorbar;
    set(gca,'ZTick',zticks,'ZTickLabel',zticklbl);
    grid on; box on
    xlabel('\tau'); ylabel('\rho'); zlabel('Median Best (log)');
    title(sprintf('F%d 3D Surface', F), 'FontWeight','normal');
    view([-38 32]); axis tight;

    saveas(fig, sprintf('%s-F%02d-Grid3D.png', Tag, F));
    close(fig);
end

save(sprintf('results_%s_compare.mat', Tag), 'All','algoMode','nPop','dim','MaxIt','run_times', ...
     'tau_grid','rho_grid','K_fixed','steps_km','pick_rule','master_seed','skipFuncs');
fprintf('\nSaved results to results_%s_compare.mat\n', Tag);
end

%% ================== Helpers & runners ==================
function [bestScore, curve] = run_baseline(mode, nPop, MaxIt, lb, ub, dim, fobj)
    switch upper(mode)
        case 'PSO'
            [bestScore, ~, curve] = PSO(nPop, MaxIt, lb, ub, dim, fobj);
        case 'DE'
            [bestScore, ~, curve] = DE (nPop, MaxIt, lb, ub, dim, fobj);
        otherwise
            error('Unknown mode: %s', mode);
    end
    curve = curve(:)';   % ensure row
end

function [bestScore, curve] = run_pso_cce(nPop, MaxIt, lb, ub, dim, fobj, cce_opts)
    Vmax = 2; noP = nPop;
    wMax = 0.9; wMin = 0.2; c1 = 2; c2 = 2;
    if isscalar(lb), lb = lb*ones(1,dim); end
    if isscalar(ub), ub = ub*ones(1,dim); end

    vel        = zeros(noP, dim);
    pBestScore = inf(noP, 1);
    pBest      = zeros(noP, dim);
    gBest      = zeros(1, dim);
    curve      = zeros(1, MaxIt);
    pos        = initialization(noP, dim, ub, lb);
    gBestScore = inf;

    cce_state = struct('t',1);

    for l = 1:MaxIt
        fvals = zeros(noP,1);
        % evaluate + bests
        for i = 1:noP
            Flag4ub  = pos(i,:)>ub; Flag4lb = pos(i,:)<lb;
            pos(i,:) = (pos(i,:).*(~(Flag4ub+Flag4lb))) + ub.*Flag4ub + lb.*Flag4lb;
            f = fobj(pos(i,:)); fvals(i)=f;
            if f < pBestScore(i), pBestScore(i)=f; pBest(i,:)=pos(i,:); end
            if f < gBestScore, gBestScore=f; gBest=pos(i,:); end
        end
        % PSO update
        w = wMax - l * ((wMax - wMin) / MaxIt);
        for i = 1:noP
            r1 = rand(1,dim); r2 = rand(1,dim);
            vel(i,:) = w*vel(i,:) + c1*r1.*(pBest(i,:)-pos(i,:)) + c2*r2.*(gBest-pos(i,:));
            vel(i,:) = max(min(vel(i,:),Vmax),-Vmax);
            pos(i,:) = pos(i,:) + vel(i,:);
        end
        % CCE plug-in
        [pos_new, cce_state, info] = CCE(pos, fvals, lb, ub, cce_state, cce_opts);
        if info.did_eliminate
            idx = info.replaced_idx(:)';
            vel(idx,:)      = 0;
            pBest(idx,:)    = pos_new(idx,:);
            pBestScore(idx) = inf;
        end
        pos = pos_new;

        curve(l) = gBestScore;
    end
    bestScore = gBestScore;
end

function [bestScore, curve] = run_de_cce(nPop, MaxIt, lb, ub, dim, fobj, cce_opts)
    VarSize = [1 dim];
    if isscalar(lb), lb = repmat(lb,1,dim); end
    if isscalar(ub), ub = repmat(ub,1,dim); end
    lo = lb; hi = ub; range = hi - lo;

    F  = 0.5; pCR = 0.9;

    BestSol.Position = []; BestSol.Cost = inf;
    pop = repmat(struct('Position',[],'Cost',inf), nPop, 1);
    for i=1:nPop
        pop(i).Position = lo + rand(VarSize).*range;
        pop(i).Cost     = fobj(pop(i).Position);
        if pop(i).Cost < BestSol.Cost, BestSol = pop(i); end
    end
    curve = zeros(1, MaxIt);

    cce_state = struct('t',1);

    for it=1:MaxIt
        for i=1:nPop
            x = pop(i).Position;
            A = randperm(nPop); A(A==i)=[];
            a=A(1); b=A(2); c=A(3);
            v = pop(a).Position + F*(pop(b).Position - pop(c).Position);
            v = min(max(v, lo), hi);

            z  = x;
            j0 = randi([1 numel(x)]);
            for j=1:numel(x)
                if j==j0 || rand<=pCR, z(j)=v(j); else, z(j)=x(j); end
            end

            New.Position = z;
            New.Cost     = fobj(z);

            if New.Cost < pop(i).Cost
                pop(i) = New;
                if New.Cost < BestSol.Cost, BestSol = New; end
            end
        end

        % CCE plug-in (after selection)
        X = cat(1,pop.Position);
        f = [pop.Cost]';
        [X2, cce_state, info] = CCE(X, f, lo, hi, cce_state, cce_opts);
        if info.did_eliminate
            idx = info.replaced_idx(:)';
            for t = 1:numel(idx)
                ii = idx(t);
                pop(ii).Position = X2(ii,:);
                pop(ii).Cost     = inf;   % next generation will evaluate
            end
        end

        curve(it) = BestSol.Cost;
    end
    bestScore = BestSol.Cost;
end
