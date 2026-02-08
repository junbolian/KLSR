function main_KLSR_compare
% ============================================================
%  Baseline (PSO/CSO/GA/JADE) vs KLSR
%  —— 仅扫描两个最重要的参数：tau（拟合周期） × p0（初始触发概率）
% ============================================================

clc; clear; close all; warning off
                                                                                                                                                                                                                                                                                                                                                                                                        
%% ------------- Choose algorithm family -------------
algoMode  = 'PSO';     % choose from: 'PSO', 'CSO', 'GA', 'JADE'

%% ------------- Experiment setup -------------
nPop      = 100;
dim       = 100;
run_times = 10;                 % 取中位数更稳
FuncList  = 1:30;               % CEC-2017 F1..F30
skipFuncs = [2];                % 可按需移除

% ------- 只扫描两个参数 -------
tau_grid  = [25 50 100];        % τ：拟合周期
p0_grid   = [0.2 0.3 0.5];      % 初始触发概率 p0
p1_fixed  = 0.10;               % 末期触发概率固定
stall_T   = 7;                  % 其它参数固定（不参与扫描）
only_on_stall = true; 
bound_mode    = 'clip';

% 固定的拟合与触发默认（不进入网格）
fit_opts_default = struct('D',128,'sigma',1.0,'lambda',1e-3,...
                          'k',min(3*dim,200),'df',inf,'quality_min',0.6);
quota_fixed   = 0.10;   % 每代最多 10% 个体触发
arch_max_fixed= 500;
stallG_fixed  = 20;     % 全局停滞门控
alpha_fixed   = [1 0.5 0.25];
maxeval_fixed = 1;

% Aesthetics
set(0,'DefaultAxesFontName','Helvetica','DefaultTextFontName','Helvetica');
convCols = [0.15 0.15 0.15; 0.00 0.45 0.74];
try, cmap = turbo(256); catch, cmap = parula(256); end

% Reproducibility
master_seed = 20250810;

Tag = sprintf('KLSR_%s_CEC2017_D%d', upper(algoMode), dim);
All = struct();

for F = FuncList
    if ismember(F, skipFuncs)
        fprintf('\n========== %s  F%02d  (dim=%d) ==========\n', upper(algoMode), F, dim);
        fprintf('>> Skipping F%02d as requested.\n', F);
        continue;
    end

    % 获取函数与一致的迭代预算
    [lb, ub, dim_true, fobj] = Get_Functions_cec2017(F, dim);
    MaxIt = floor((10000*dim_true)/nPop);
    fprintf('\n========== %s  F%02d  (dim=%d, MaxIt=%d) ==========\n', upper(algoMode), F, dim_true, MaxIt);

    % 固定 run 的随机种子，保证公平
    rng(master_seed + F);
    base_seeds = randi(2^31-1, run_times, 1);

    %% -------- Baseline runs --------
    baseScores   = zeros(run_times,1);
    baseCurveSum = zeros(1, MaxIt);
    baseTimes    = zeros(run_times,1);

    for r = 1:run_times
        rng(base_seeds(r));
        t0 = tic;
        if strcmpi(algoMode,'PSO')
            [bestScore, ~, curve] = PSO(nPop, MaxIt, lb, ub, dim_true, fobj);
        elseif strcmpi(algoMode,'GA')
            [bestScore, ~, curve] = GA(nPop, MaxIt, lb, ub, dim_true, fobj);
        elseif strcmpi(algoMode,'JADE')
            [bestScore, ~, curve] = JADE(nPop, MaxIt, lb, ub, dim_true, fobj);
        elseif strcmpi(algoMode,'CSO')
            [bestScore, ~, curve] = CSO(nPop, MaxIt, lb, ub, dim_true, fobj);
        else
            error('Unsupported algoMode: %s', algoMode);
        end
        baseTimes(r)  = toc(t0);
        baseScores(r) = bestScore;
        baseCurveSum  = baseCurveSum + curve(:).';
    end
    baseCurveAvg = baseCurveSum / run_times;
    baseMed      = median(baseScores);
    baseMeanTime = mean(baseTimes);
    fprintf('Baseline:  median=%.3e  meanTime=%.2fs/run\n', baseMed, baseMeanTime);

    %% -------- Grid search over (tau, p0) --------
    A = numel(tau_grid); B = numel(p0_grid);
    medGrid   = nan(A,B);               % rows: tau, cols: p0
    timeGrid  = nan(A,B);
    Scores    = cell(A,B);              
    CurveSum  = cell(A,B);              

    for ia = 1:A
        for ib = 1:B
            % 本次组合的 kopt（只变 tau 和 p0）
            kopt = struct('tau',tau_grid(ia), 'p0',p0_grid(ib), 'p1',p1_fixed, ...
                          'stall_T',stall_T, 'only_on_stall',only_on_stall, ...
                          'bound_mode',bound_mode, 'fit',fit_opts_default, ...
                          'quota',quota_fixed, 'arch_max',arch_max_fixed, ...
                          'stallG',stallG_fixed, 'alpha',alpha_fixed, 'max_evals',maxeval_fixed);

            bestVec  = zeros(run_times,1);
            curveSum = zeros(1, MaxIt);
            t1 = tic;
            for r = 1:run_times
                rng(base_seeds(r));
                if strcmpi(algoMode,'PSO')
                    [bestScore, ~, curve] = PSO_KLSR(nPop, MaxIt, lb, ub, dim_true, fobj, kopt);
                elseif strcmpi(algoMode,'GA')
                    [bestScore, ~, curve] = GA_KLSR(nPop, MaxIt, lb, ub, dim_true, fobj, kopt);
                elseif strcmpi(algoMode,'JADE')
                    [bestScore, ~, curve] = JADE_KLSR(nPop, MaxIt, lb, ub, dim_true, fobj, kopt);
                elseif strcmpi(algoMode,'CSO')
                    [bestScore, ~, curve] = CSO_KLSR(nPop, MaxIt, lb, ub, dim_true, fobj, kopt);
                else
                    error('Unsupported algoMode: %s', algoMode);
                end
                bestVec(r)  = bestScore;
                curveSum    = curveSum + curve(:).';
            end
            medGrid(ia,ib)  = median(bestVec);
            timeGrid(ia,ib) = toc(t1)/run_times;
            Scores{ia,ib}   = bestVec;
            CurveSum{ia,ib} = curveSum;

            fprintf('Grid  tau=%d  p0=%.2f  median=%.3e  time=%.2fs/run\n', ...
                tau_grid(ia), p0_grid(ib), medGrid(ia,ib), timeGrid(ia,ib));
        end
    end

    % 选择网格最优
    [bestMed, idx] = min(medGrid(:));
    [ia_best, ib_best] = ind2sub(size(medGrid), idx);
    best_tau = tau_grid(ia_best);
    best_p0  = p0_grid(ib_best);
    fprintf('>> Grid-best KLSR on F%02d: tau=%d, p0=%.2f, median=%.3e\n', ...
            F, best_tau, best_p0, bestMed);

    %% -------- 复用最佳组合缓存 --------
    klsrScores   = Scores{ia_best, ib_best};
    klsrCurveAvg = CurveSum{ia_best, ib_best} / run_times;
    klsrMed      = medGrid(ia_best, ib_best);
    klsrMeanTime = timeGrid(ia_best, ib_best);
    fprintf('KLSR(best, cached): median=%.3e  meanTime=%.2fs/run\n', klsrMed, klsrMeanTime);

    %% -------- Store --------
    All(F).F              = F;
    All(F).algo           = upper(algoMode);
    All(F).baseline.med   = baseMed;
    All(F).baseline.scores= baseScores;
    All(F).baseline.curve = baseCurveAvg;
    All(F).baseline.time  = baseTimes;
    All(F).grid.tau_grid  = tau_grid;
    All(F).grid.p0_grid   = p0_grid;
    All(F).grid.medGrid   = medGrid;
    All(F).grid.timeGrid  = timeGrid;
    All(F).best.tau       = best_tau;
    All(F).best.p0        = best_p0;
    All(F).best.med       = bestMed;
    All(F).klsr.med       = klsrMed;
    All(F).klsr.scores    = klsrScores;
    All(F).klsr.curve     = klsrCurveAvg;
    All(F).klsr.time      = klsrMeanTime;

    %% -------- Plots: Convergence & Boxplot --------
    fig = figure('Color','w'); hold on
    semilogy(baseCurveAvg, 'LineWidth', 2.0, 'Color', convCols(1,:));
    semilogy(klsrCurveAvg, 'LineWidth', 2.0, 'Color', convCols(2,:));
    grid on; box on
    xlabel('Iteration'); ylabel('Best Fitness (log)');
    title(sprintf('%s  F%02d (Dim=%d)', upper(algoMode), F, dim_true), 'FontWeight','normal');
    legend({'Baseline', sprintf('KLSR(\\tau=%d, p_0=%.2f)', best_tau, best_p0)}, 'Location','northeast');
    saveas(fig, sprintf('%s-F%02d-Conv.png', Tag, F));
    close(fig);

    fig = figure('Color','w','Position',[420 240 360 220]);
    boxplot([baseScores, klsrScores], 'Labels', {'Baseline','KLSR(best)'}); 
    set(gca,'YScale','log'); grid on; box on
    ylabel('Best Fitness (log)');
    title(sprintf('F%02d (Dim=%d)', F, dim_true), 'FontWeight','normal');
    saveas(gcf, sprintf('%s-F%02d-Box.png', Tag, F));
    close(gcf);

    %% -------- 3D: tau × p0 → log10(median best) --------
    pAxis = [0, p0_grid];     % baseline at p0=0
    tauAxis = tau_grid;

    medMat = nan(numel(pAxis), numel(tauAxis));  % rows: p0, cols: tau
    medMat(1, :)     = baseMed;                  % baseline 复制到每个 tau
    medMat(2:end, :) = medGrid.';                % 转置：行=p0, 列=tau

    Z = log10(max(medMat, eps));
    zmin = floor(min(Z(:))); zmax = ceil(max(Z(:)));
    zticks   = linspace(zmin, zmax, max(3, min(7, zmax - zmin + 1)));
    zticklbl = arrayfun(@(v) sprintf('10^{%.0f}', v), zticks, 'uni', 0);

    fig = figure('Color','w','Position',[180 90 980 430]);

    % ---- 3D Bar
    subplot(1,2,1);
    bh = bar3(Z, 'detached');
    for k = 1:numel(bh), set(bh(k),'EdgeColor',[0.35 0.35 0.35]); end
    colormap(cmap); caxis([zmin zmax]); colorbar;
    grid on; box on
    set(gca,'ZTick',zticks,'ZTickLabel',zticklbl);
    set(gca,'XTick',1:numel(tauAxis), 'XTickLabel', arrayfun(@(x)sprintf('%d',x),tauAxis,'uni',0));
    set(gca,'YTick',1:numel(pAxis),  'YTickLabel', arrayfun(@(y)sprintf('%.2f',y),pAxis,'uni',0));
    xlabel('\\tau (fit period)'); ylabel('p_0 (trigger prob)'); zlabel('Median Best (log)');
    title(sprintf('F%d 3D Bar (Baseline at p_0=0)', F), 'FontWeight','normal');
    view([-40 28]); axis tight;

    % ---- 3D Surface
    subplot(1,2,2);
    [TAU, P0] = meshgrid(tauAxis, pAxis);
    surf(TAU, P0, Z, 'EdgeAlpha',0.2, 'FaceAlpha',0.95);
    colormap(cmap); caxis([zmin zmax]); colorbar;
    set(gca,'ZTick',zticks,'ZTickLabel',zticklbl);
    grid on; box on
    xlabel('\\tau (fit period)'); ylabel('p_0 (trigger prob)'); zlabel('Median Best (log)');
    title(sprintf('F%d 3D Surface', F), 'FontWeight','normal');
    view([-38 32]); axis tight;

    saveas(fig, sprintf('%s-F%02d-Grid3D.png', Tag, F));
    close(fig);
end

save(sprintf('results_%s_compare.mat', Tag), 'All','algoMode','nPop','dim','run_times', ...
     'tau_grid','p0_grid','p1_fixed','stall_T','only_on_stall','master_seed','skipFuncs','fit_opts_default');
fprintf('\nSaved results to results_%s_compare.mat\n', Tag);
end
