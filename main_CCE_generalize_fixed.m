function main_CCE_generalize_fixed
% ============================================================
%  CCE generalization (fixed params) across PSO / DE / GA / SHADE / JADE
%  - Baseline vs CCE, 20 runs each, CEC-2017, D=100, budget = 10000*D
%  - Saves convergence curves, combined boxplots, and LaTeX tables
%  - Author: Junbo Jacob Lian
% ============================================================

clc; clear; close all; warning off
rng('shuffle');

%% ---------- Fixed CCE parameters (edit here if needed) ----------
global CCE_OPTS
CCE_OPTS = struct('tau',2, ...          % refresh interval
                  'rho',0.9, ...        % fraction replaced in the worst cluster
                  'K',[], ...           % [] => auto-K = min(8, ceil(sqrt(N)))
                  'steps',5, ...        % Lloyd steps in k-means
                  'pick','min', ...     % 'min' or 'median'
                  'avoid_best',true, ...
                  'keep_one',true);

%% ---------- Experiment setup ----------
nPop      = 100;
dim       = 100;
MaxIt     = (10000*dim)/nPop;
run_times = 15;

FuncList  = 1:30;         % CEC-2017: F1..F30
skipFuncs = [2];          % e.g., F2 skipped in your environment

% If your CEC path is elsewhere, add it here:
% addpath('E:\Optimization\IKUN\code');

% aesthetics
set(0,'DefaultAxesFontName','Helvetica','DefaultTextFontName','Helvetica');
% Colors for 3 algorithms (baseline vs CCE pairs)
cols = [0.3 0.3 0.3; 0.85 0.33 0.10;      % DE: gray, orange
        0.5 0.5 0.5; 0.49 0.18 0.56;      % SHADE: medium gray, purple
        0.6 0.6 0.6; 0.30 0.75 0.93];     % JADE: light gray, cyan

% output folder
Tag   = sprintf('GEN_CCE_FIXED_D%d', dim);
outDir = ['./', Tag];
if ~exist(outDir,'dir'), mkdir(outDir); end

%% ---------- Algorithm roster (baseline vs CCE) ----------
Algos = { ...
  'DE',    @DE,      @CCEDE  ; ...
  'SHADE', @SHADE,   @CCESHADE ; ...
  'JADE',  @JADE,    @CCEJADE  ...
};

%% ---------- Initialize LaTeX table ----------
latexFile = fullfile(outDir, 'results_table.tex');
fid = fopen(latexFile, 'w');
fprintf(fid, '\\begin{table}[t]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Performance comparison of baseline vs CCE-enhanced algorithms on CEC-2017 (D=%d, %d runs)}\n', dim, run_times);
fprintf(fid, '\\label{tab:cce-generalize}\n');
fprintf(fid, '\\begin{tabular}{l|cc|cc|cc|cc|cc|cc}\n');
fprintf(fid, '\\toprule\n');
fprintf(fid, 'Function');
for a = 1:size(Algos,1)
    fprintf(fid, ' & \\multicolumn{2}{c|}{%s}', Algos{a,1});
end
fprintf(fid, ' \\\\\n');
fprintf(fid, ' ');
for a = 1:size(Algos,1)
    fprintf(fid, ' & Base & CCE');
end
fprintf(fid, ' \\\\\n');
fprintf(fid, '\\midrule\n');

%% ---------- Main loop ----------
All = struct();
for F = FuncList
    if ismember(F, skipFuncs)
        fprintf('>> Skip F%02d.\n', F);
        continue;
    end
    [lb, ub, dim_true, fobj] = Get_Functions_cec2017(F, dim);
    fprintf('\n========= CEC2017 F%02d  (D=%d) =========\n', F, dim_true);
    
    % Store all scores for combined boxplot
    allScores = [];
    groupLabels = {};
    
    % Start LaTeX row
    fprintf(fid, 'F%02d', F);
    
    for a = 1:size(Algos,1)
        name       = Algos{a,1};
        baselineFn = Algos{a,2};
        cceFn      = Algos{a,3};
        fprintf('--- %s ---\n', name);

        % ---------- Baseline ----------
        baseScores   = zeros(run_times,1);
        baseTimes    = zeros(run_times,1);
        baseCurveSum = zeros(1, MaxIt);
        for r = 1:run_times
            t0 = tic;
            [bestScore, ~, curve] = baselineFn(nPop, MaxIt, lb, ub, dim_true, fobj);
            baseTimes(r)  = toc(t0);
            baseScores(r) = bestScore;
            baseCurveSum  = baseCurveSum + curve(:)';
        end
        baseMed      = median(baseScores);
        baseStd      = std(baseScores);
        baseMeanTime = mean(baseTimes);
        baseCurveAvg = baseCurveSum / run_times;
        fprintf('Baseline: median=%.3e  std=%.3e  meanTime=%.2fs/run\n', baseMed, baseStd, baseMeanTime);

        % ---------- CCE (fixed params via global CCE_OPTS) ----------
        cceScores   = zeros(run_times,1);
        cceTimes    = zeros(run_times,1);
        cceCurveSum = zeros(1, MaxIt);
        for r = 1:run_times
            t1 = tic;
            [bestScore, ~, curve] = cceFn(nPop, MaxIt, lb, ub, dim_true, fobj);
            cceTimes(r)  = toc(t1);
            cceScores(r) = bestScore;
            cceCurveSum  = cceCurveSum + curve(:)';
        end
        cceMed      = median(cceScores);
        cceStd      = std(cceScores);
        cceMeanTime = mean(cceTimes);
        cceCurveAvg = cceCurveSum / run_times;
        fprintf('   CCE:   median=%.3e  std=%.3e  meanTime=%.2fs/run\n', cceMed, cceStd, cceMeanTime);

        % ---------- Statistical comparison ----------
        [p, h] = ranksum(baseScores, cceScores);  % Wilcoxon rank-sum test
        if h == 1  % Significant difference
            if cceMed < baseMed
                compSymbol = '+';  % CCE wins
            else
                compSymbol = '-';  % Baseline wins
            end
        else
            compSymbol = '=';  % No significant difference
        end

        % ---------- Save to struct ----------
        All(F).(name).baseline.scores = baseScores;
        All(F).(name).baseline.curve  = baseCurveAvg;
        All(F).(name).baseline.time   = baseTimes;
        All(F).(name).baseline.median = baseMed;
        All(F).(name).baseline.std    = baseStd;
        All(F).(name).CCE.scores      = cceScores;
        All(F).(name).CCE.curve       = cceCurveAvg;
        All(F).(name).CCE.time        = cceTimes;
        All(F).(name).CCE.median      = cceMed;
        All(F).(name).CCE.std         = cceStd;
        All(F).(name).comparison       = compSymbol;
        All(F).(name).pvalue           = p;
        All(F).(name).meta.dim         = dim_true;
        
        % Add to combined data
        allScores = [allScores, baseScores', cceScores'];
        groupLabels = [groupLabels, {[name '-Base']}, {[name '-CCE']}];
        
        % Write to LaTeX table
        fprintf(fid, ' & %.2e & \\textbf{%.2e}%s', baseMed, cceMed, compSymbol);
        
        % ---------- Individual convergence plot ----------
        fig1 = figure('Color','w'); hold on
        semilogy(baseCurveAvg, 'LineWidth',2.0, 'Color',cols(2*a-1,:));
        semilogy(cceCurveAvg,  'LineWidth',2.0, 'Color',cols(2*a,:));
        grid on; box on
        xlabel('Iteration'); ylabel('Best Fitness (log)');
        title(sprintf('%s  F%02d  (D=%d)', name, F, dim_true), 'FontWeight','normal');
        legend({[name '-Baseline'],[name '-CCE']}, 'Location','northeast');
        saveas(fig1, fullfile(outDir, sprintf('%s-F%02d-Conv.png', name, F)));
        close(fig1);
    end
    
    fprintf(fid, ' \\\\\n');
    
    % ---------- Combined boxplot for all algorithms ----------
    fig2 = figure('Color','w','Position',[100 100 800 400]);
    boxData = [];
    boxGroup = [];
    for i = 1:length(allScores)/run_times
        boxData = [boxData; allScores((i-1)*run_times+1:i*run_times)'];
        boxGroup = [boxGroup; i*ones(run_times,1)];
    end
    
    boxplot(boxData, boxGroup, 'Labels', groupLabels);
    set(gca,'YScale','log'); 
    grid on; box on
    ylabel('Best Fitness (log)');
    title(sprintf('F%02d  (D=%d) - All Algorithms Comparison', F, dim_true), 'FontWeight','normal');
    xtickangle(45);
    
    % Color the boxes
    h = findobj(gca,'Tag','Box');
    for j=1:length(h)
        idx = length(h) - j + 1;
        patch(get(h(j),'XData'), get(h(j),'YData'), cols(idx,:), 'FaceAlpha',.5);
    end
    
    saveas(fig2, fullfile(outDir, sprintf('F%02d-AllAlgos-Box.png', F)));
    close(fig2);
end

% Complete LaTeX table
fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\end{table}\n\n');

% Summary statistics table
fprintf(fid, '\\begin{table}[t]\n');
fprintf(fid, '\\centering\n');
fprintf(fid, '\\caption{Summary statistics: Win/Tie/Loss counts for CCE vs baseline}\n');
fprintf(fid, '\\label{tab:cce-summary}\n');
fprintf(fid, '\\begin{tabular}{lccc}\n');
fprintf(fid, '\\toprule\n');
fprintf(fid, 'Algorithm & Win (+) & Tie (=) & Loss (-) \\\\\n');
fprintf(fid, '\\midrule\n');

% Count wins/ties/losses for each algorithm
for a = 1:size(Algos,1)
    name = Algos{a,1};
    wins = 0; ties = 0; losses = 0;
    
    for F = FuncList
        if ismember(F, skipFuncs) || ~isfield(All(F), name)
            continue;
        end
        
        comp = All(F).(name).comparison;
        if strcmp(comp, '+')
            wins = wins + 1;
        elseif strcmp(comp, '=')
            ties = ties + 1;
        else
            losses = losses + 1;
        end
    end
    
    fprintf(fid, '%s & %d & %d & %d \\\\\n', name, wins, ties, losses);
end

fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\end{tabular}\n');
fprintf(fid, '\\end{table}\n');

fclose(fid);

save(fullfile(outDir, sprintf('results_%s.mat', Tag)), ...
     'All','Algos','nPop','dim','MaxIt','run_times','CCE_OPTS','FuncList','skipFuncs');
fprintf('\nSaved results to %s\n', fullfile(outDir, sprintf('results_%s.mat', Tag)));
fprintf('LaTeX tables saved to %s\n', latexFile);

%% ---------- Generate overall comparison plot ----------
figure('Color','w','Position',[100 100 1400 900]);
nFuncs = length(FuncList) - length(skipFuncs);
nAlgos = size(Algos,1);
plotIdx = 1;

for fIdx = 1:length(FuncList)
    F = FuncList(fIdx);
    if ismember(F, skipFuncs)
        continue;
    end
    
    subplot(6, 5, plotIdx);
    
    % Prepare data for this function
    algoData = [];
    for a = 1:nAlgos
        name = Algos{a,1};
        if isfield(All(F), name)
            algoData = [algoData, All(F).(name).baseline.scores', All(F).(name).CCE.scores'];
        end
    end
    
    % Mini boxplot
    boxplot(algoData, 'Labels', repmat({'B','C'}, 1, nAlgos));
    set(gca, 'YScale', 'log', 'FontSize', 8);
    ylabel(sprintf('F%02d', F), 'FontSize', 10);
    grid on;
    
    % Add algorithm names at the bottom
    if plotIdx > 25  % Last row
        xticks(1.5:2:2*nAlgos);
        xticklabels(Algos(:,1));
        xtickangle(45);
    else
        set(gca, 'XTickLabel', []);
    end
    
    plotIdx = plotIdx + 1;
end

sgtitle('CCE Performance Across All Functions and Algorithms', 'FontSize', 14);
saveas(gcf, fullfile(outDir, 'All_Functions_Comparison.png'));

fprintf('\n============ Experiment Complete ============\n');
end