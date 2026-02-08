function summary_all()
clc; close all;

% --- locate run directory robustly ---
runDir = fileparts(mfilename('fullpath'));
if isempty(runDir), runDir = pwd; end

resultsDir = fullfile(runDir, 'results');
if ~exist(resultsDir, 'dir'), mkdir(resultsDir); end

% ===== scenarios to run =====
% IMPORTANT: matRel must match what each run_uav_30runs_S*.m saves
jobs = {
    'S1', 'run_uav_30runs_S1.m', 'uav_30runs_S1.mat';
    'S2', 'run_uav_30runs_S2.m', 'uav_30runs_S2.mat';
    'S3', 'run_uav_30runs_S3.m', 'uav_30runs_S3.mat'
};

pairs = {'CSO','CSO_KLSR'; 'JADE','JADE_KLSR'; 'GA','GA_KLSR'; 'PSO','PSO_KLSR'};

AllMeanStd = table();
AllPvalues = table();

for j = 1:size(jobs,1)
    tag    = jobs{j,1};
    script = jobs{j,2};
    matRel = jobs{j,3};

    fprintf('\n==================== %s ====================\n', tag);
    fprintf('Running (base workspace): %s\n', script);

    scriptPath = fullfile(runDir, script);

    % --- Run scenario in BASE workspace so its clear/clearvars won't touch this function vars ---
    evalin('base', sprintf('run(''%s'');', scriptPath));

    % --- Load the MAT produced by scenario ---
    srcMat = fullfile(runDir, matRel);
    if ~exist(srcMat, 'file')
        error('[%s] Missing mat file: %s\nCheck that %s saves to this name.', tag, srcMat, script);
    end

    % archive (avoid overwrite)
    dstMat = fullfile(resultsDir, sprintf('uav_30runs_%s.mat', tag));
    copyfile(srcMat, dstMat);

    % --- compute mean/std/time + rank + pvalue ---
    S = load(dstMat);  % expects: final_best, runtime_s, methods
    methods    = S.methods;
    final_best = S.final_best;
    runtime_s  = S.runtime_s;

    mean_best = mean(final_best, 1);
    std_best  = std(final_best, 0, 1);
    mean_time = mean(runtime_s, 1);

    [~, ord] = sort(mean_best, 'ascend');   % smaller is better
    rankv = zeros(size(mean_best));
    rankv(ord) = 1:numel(mean_best);

    T = table( ...
        repmat({tag}, numel(methods), 1), methods(:), ...
        mean_best(:), std_best(:), mean_time(:), rankv(:), ...
        'VariableNames', {'Scenario','Method','MeanBest','StdBest','MeanTime_s','Rank'} ...
    );
    AllMeanStd = [AllMeanStd; T]; %#ok<AGROW>

    % paired Wilcoxon signed-rank on per-run deltas (Baseline - KLSR)
    for i = 1:size(pairs,1)
        baseName = pairs{i,1};
        klsrName = pairs{i,2};

        b = find(strcmp(methods, baseName), 1);
        k = find(strcmp(methods, klsrName), 1);
        if isempty(b) || isempty(k)
            warning('[%s] Missing method: %s or %s', tag, baseName, klsrName);
            continue;
        end

        delta = final_best(:, b) - final_best(:, k); % >0 means KLSR better
        p = NaN;
        try
            p = signrank(delta);
        catch
            % if stats toolbox missing, keep NaN
        end

        Tp = table( ...
            {tag}, {baseName}, {klsrName}, mean(delta), p, ...
            'VariableNames', {'Scenario','Base','KLSR','MeanDelta','PValue'} ...
        );
        AllPvalues = [AllPvalues; Tp]; %#ok<AGROW>
    end

    fprintf('Archived: %s\n', dstMat);
end

writetable(AllMeanStd, fullfile(resultsDir, 'scenario_mean_std_rank.csv'));
writetable(AllPvalues, fullfile(resultsDir, 'pair_pvalues.csv'));

disp('==================== DONE ====================');
disp('Saved: results/scenario_mean_std_rank.csv');
disp('Saved: results/pair_pvalues.csv');
end
