function main_KLSR_master()
% MAIN_KLSR_MASTER
% Unified entry point for KLSR experiments (CSO/GA/PSO/JADE).
% Supports:
%   - mode = 'sig'     : minimal outputs for stats (scores, medians)
%   - mode = 'compare' : full outputs (scores, time, curve, grid)
%
% Output files:
%   results_KLSR_<ALGO>_CEC2017_D<dim>_<mode>.mat
%
% NOTE: This script does NOT run cross-algorithm comparison in one file.
% It runs one algorithm per call (or a list via algoList).

clc; clear; close all; warning off;

%% ---------------- User options ----------------
algoList = {'CSO','GA','PSO','JADE'};  % only keep these four
run_single_algo = '';                 % set to 'CSO'/'GA'/'PSO'/'JADE' to run only one
dims     = [30, 50];                   % dimensions to run
mode     = 'sig';                 % 'sig' or 'compare'
save_plots = false;                    % only works in 'compare' mode

% Experiment setup
nPop      = 100;
run_times = 10;
FuncList  = 1:30;
skipFuncs = [2];

% KLSR parameters (default from your sensitivity)
tau_grid  = [25 50 100];
p0_grid   = [0.2 0.3 0.5];
p1_fixed  = 0.10;
stall_T   = 7;
only_on_stall = true;
bound_mode = 'clip';

fit_opts_default = struct('D',128,'sigma',1.0,'lambda',1e-3, ...
                          'k',min(3*50,200),'df',inf,'quality_min',0.6);
quota_fixed   = 0.10;
arch_max_fixed= 500;
stallG_fixed  = 20;
alpha_fixed   = [1 0.5 0.25];
maxeval_fixed = 1;

master_seed = 20250810;

% Aesthetics (for plots)
if save_plots
    set(0,'DefaultAxesFontName','Helvetica','DefaultTextFontName','Helvetica');
    convCols = [0.15 0.15 0.15; 0.00 0.45 0.74];
    try, cmap = turbo(256); catch, cmap = parula(256); end
end

%% ---------------- Run ----------------
if ~isempty(run_single_algo)
    algoList = {upper(run_single_algo)};
end

for di = 1:numel(dims)
    dim = dims(di);

    % update k (depends on dim)
    fit_opts_default.k = min(3*dim, 200);

    for ai = 1:numel(algoList)
        algoMode = upper(algoList{ai});

        Tag = sprintf('KLSR_%s_CEC2017_D%d', algoMode, dim);
        All = struct();

        fprintf('============================================================\n');
        fprintf('Algo=%s | D=%d | N=%d | R=%d | mode=%s\n', algoMode, dim, nPop, run_times, mode);
        fprintf('tau=%s | p0=%s | p1=%.2f\n', mat2str(tau_grid), mat2str(p0_grid), p1_fixed);
        fprintf('Functions: 1..30 (skip F02)\n');
        fprintf('============================================================\n\n');

        for F = FuncList
            if ismember(F, skipFuncs)
                fprintf('Skip F%02d\n', F);
                continue;
            end

            [lb, ub, dim_true, fobj] = Get_Functions_cec2017(F, dim);
            if isempty(dim_true) || dim_true ~= dim
                dim_true = dim;
            end
            MaxIt = floor((10000*dim_true)/nPop);

            rng(master_seed + F, 'twister');
            base_seeds = randi(2^31-1, run_times, 1);

            % ---- Baseline ----
            baseScores   = zeros(run_times,1);
            baseCurveSum = zeros(1, MaxIt);
            baseTimes    = zeros(run_times,1);

            for r = 1:run_times
                rng(base_seeds(r), 'twister');
                t0 = tic;
                [bestScore, ~, curve] = run_baseline_algo(algoMode,nPop,MaxIt,lb,ub,dim_true,fobj);
                baseTimes(r)  = toc(t0);
                baseScores(r) = bestScore;
                baseCurveSum  = baseCurveSum + curve(:).';
            end
            baseCurveAvg = baseCurveSum / run_times;
            baseMed = median(baseScores);

            % ---- Grid Search ----
            A = numel(tau_grid); B = numel(p0_grid);
            medGrid  = nan(A,B);
            timeGrid = nan(A,B);
            Scores   = cell(A,B);
            CurveSum = cell(A,B);

            for ia = 1:A
                for ib = 1:B
                    kopt = struct('tau',tau_grid(ia),'p0',p0_grid(ib),'p1',p1_fixed, ...
                                  'stall_T',stall_T,'only_on_stall',only_on_stall, ...
                                  'bound_mode',bound_mode,'fit',fit_opts_default, ...
                                  'quota',quota_fixed,'arch_max',arch_max_fixed, ...
                                  'stallG',stallG_fixed,'alpha',alpha_fixed,'max_evals',maxeval_fixed);

                    bestVec  = zeros(run_times,1);
                    curveSum = zeros(1, MaxIt);

                    t1 = tic;
                    for r = 1:run_times
                        rng(base_seeds(r), 'twister');
                        [bestScore, ~, curve] = run_klsr_algo(algoMode,nPop,MaxIt,lb,ub,dim_true,fobj,kopt);
                        bestVec(r) = bestScore;
                        curveSum   = curveSum + curve(:).';
                    end

                    medGrid(ia,ib)  = median(bestVec);
                    timeGrid(ia,ib) = toc(t1)/run_times;
                    Scores{ia,ib}   = bestVec;
                    CurveSum{ia,ib} = curveSum;
                end
            end

            [bestMed, idx] = min(medGrid(:));
            [ia_best, ib_best] = ind2sub(size(medGrid), idx);

            klsrScores   = Scores{ia_best, ib_best};
            klsrCurveAvg = CurveSum{ia_best, ib_best} / run_times;
            klsrMeanTime = timeGrid(ia_best, ib_best);

            % ---- Store ----
            All(F).F = F;
            All(F).algo = algoMode;

            All(F).baseline.med    = baseMed;
            All(F).baseline.scores = baseScores;

            All(F).klsr.med        = bestMed;
            All(F).klsr.scores     = klsrScores;

            if strcmpi(mode, 'compare')
                All(F).baseline.time  = baseTimes;
                All(F).baseline.curve = baseCurveAvg;
                All(F).grid.tau_grid  = tau_grid;
                All(F).grid.p0_grid   = p0_grid;
                All(F).grid.medGrid   = medGrid;
                All(F).grid.timeGrid  = timeGrid;
                All(F).best.tau       = tau_grid(ia_best);
                All(F).best.p0        = p0_grid(ib_best);
                All(F).klsr.curve     = klsrCurveAvg;
                All(F).klsr.time      = klsrMeanTime;
            end

            % ---- Plots (optional, compare mode only) ----
            if save_plots
                if ~strcmpi(mode, 'compare')
                    warning('save_plots=true but mode is not compare. Skip plots.');
                else
                    % Convergence plot
                    fig = figure('Color','w'); hold on;
                    semilogy(baseCurveAvg, 'LineWidth', 2.0, 'Color', convCols(1,:));
                    semilogy(klsrCurveAvg, 'LineWidth', 2.0, 'Color', convCols(2,:));
                    grid on; box on;
                    xlabel('Iteration'); ylabel('Best Fitness (log)');
                    title(sprintf('%s  F%02d (Dim=%d)', algoMode, F, dim_true), 'FontWeight','normal');
                    legend({'Baseline', sprintf('KLSR(\\tau=%d, p_0=%.2f)', tau_grid(ia_best), p0_grid(ib_best))}, ...
                        'Location','northeast');
                    saveas(fig, sprintf('%s-F%02d-Conv.png', Tag, F));
                    close(fig);

                    % Boxplot
                    fig = figure('Color','w','Position',[420 240 360 220]);
                    boxplot([baseScores, klsrScores], 'Labels', {'Baseline','KLSR(best)'});
                    set(gca,'YScale','log'); grid on; box on;
                    ylabel('Best Fitness (log)');
                    title(sprintf('F%02d (Dim=%d)', F, dim_true), 'FontWeight','normal');
                    saveas(fig, sprintf('%s-F%02d-Box.png', Tag, F));
                    close(fig);

                    % 3D grid (tau x p0)
                    pAxis   = [0, p0_grid];  % baseline at p0=0
                    tauAxis = tau_grid;
                    medMat = nan(numel(pAxis), numel(tauAxis));
                    medMat(1, :)     = baseMed;
                    medMat(2:end, :) = medGrid.';
                    Z = log10(max(medMat, eps));
                    zmin = floor(min(Z(:)));
                    zmax = ceil(max(Z(:)));
                    zticks   = linspace(zmin, zmax, max(3, min(7, zmax - zmin + 1)));
                    zticklbl = arrayfun(@(v) sprintf('10^{%.0f}', v), zticks, 'uni', 0);

                    fig = figure('Color','w','Position',[180 90 980 430]);
                    subplot(1,2,1);
                    bh = bar3(Z, 'detached');
                    for k = 1:numel(bh), set(bh(k),'EdgeColor',[0.35 0.35 0.35]); end
                    colormap(cmap); caxis([zmin zmax]); colorbar;
                    grid on; box on;
                    set(gca,'ZTick',zticks,'ZTickLabel',zticklbl);
                    set(gca,'XTick',1:numel(tauAxis), 'XTickLabel', arrayfun(@(x)sprintf('%d',x),tauAxis,'uni',0));
                    set(gca,'YTick',1:numel(pAxis),  'YTickLabel', arrayfun(@(y)sprintf('%.2f',y),pAxis,'uni',0));
                    xlabel('\\tau (fit period)'); ylabel('p_0 (trigger prob)'); zlabel('Median Best (log)');
                    title(sprintf('F%02d 3D Bar (Baseline at p_0=0)', F), 'FontWeight','normal');
                    view([-40 28]); axis tight;

                    subplot(1,2,2);
                    [TAU, P0] = meshgrid(tauAxis, pAxis);
                    surf(TAU, P0, Z, 'EdgeAlpha',0.2, 'FaceAlpha',0.95);
                    colormap(cmap); caxis([zmin zmax]); colorbar;
                    set(gca,'ZTick',zticks,'ZTickLabel',zticklbl);
                    grid on; box on;
                    xlabel('\\tau (fit period)'); ylabel('p_0 (trigger prob)'); zlabel('Median Best (log)');
                    title(sprintf('F%02d 3D Surface', F), 'FontWeight','normal');
                    view([-38 32]); axis tight;

                    saveas(fig, sprintf('%s-F%02d-Grid3D.png', Tag, F));
                    close(fig);
                end
            end

            fprintf('F%02d done | baseMed=%.3e | klsrBestMed=%.3e | best(tau=%d,p0=%.2f)\n', ...
                F, baseMed, bestMed, tau_grid(ia_best), p0_grid(ib_best));
        end

        outName = sprintf('results_%s_%s.mat', Tag, lower(mode));
        save(outName, 'All','algoMode','nPop','dim','run_times', ...
            'tau_grid','p0_grid','p1_fixed','stall_T','only_on_stall','master_seed','skipFuncs','fit_opts_default');
        fprintf('\nSaved: %s\n', outName);
    end
end
end

%% ================= Algorithms ===================

function [bestScore,bestX,curve] = run_baseline_algo(algoMode,nPop,MaxIt,lb,ub,dim,fobj)
switch upper(algoMode)
    case 'PSO',   [bestScore,bestX,curve]=PSO(nPop,MaxIt,lb,ub,dim,fobj);
    case 'GA',    [bestScore,bestX,curve]=GA(nPop,MaxIt,lb,ub,dim,fobj);
    case 'JADE',  [bestScore,bestX,curve]=JADE(nPop,MaxIt,lb,ub,dim,fobj);
    case 'CSO',   [bestScore,bestX,curve]=CSO(nPop,MaxIt,lb,ub,dim,fobj);
    otherwise, error('Unsupported algoMode: %s', algoMode);
end
curve = curve(:).';
end

function [bestScore,bestX,curve] = run_klsr_algo(algoMode,nPop,MaxIt,lb,ub,dim,fobj,kopt)
switch upper(algoMode)
    case 'PSO',   [bestScore,bestX,curve]=PSO_KLSR(nPop,MaxIt,lb,ub,dim,fobj,kopt);
    case 'GA',    [bestScore,bestX,curve]=GA_KLSR(nPop,MaxIt,lb,ub,dim,fobj,kopt);
    case 'JADE',  [bestScore,bestX,curve]=JADE_KLSR(nPop,MaxIt,lb,ub,dim,fobj,kopt);
    case 'CSO',   [bestScore,bestX,curve]=CSO_KLSR(nPop,MaxIt,lb,ub,dim,fobj,kopt);
    otherwise, error('Unsupported algoMode: %s', algoMode);
end
curve = curve(:).';
end
