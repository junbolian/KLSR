function fs_feature_exp(CLEAN_PATH, DATASET_NAME, CFG)
% fs_feature_exp
% Core driver: run feature-selection + ELM classification for one dataset,
% comparing {CSO, JADE, GA, PSO} and their KLSR variants.
%
% Input:
%   CLEAN_PATH   : numeric CSV, last column is label, labels are 1..K
%   DATASET_NAME : string like 'alzheimers'
%   CFG          : struct with fields N, MaxIt, runs, test_ratio, seed0
%
% Output:
%   Saves results under: run/results/DATASET_NAME/

    % -------------------- defaults --------------------
    if nargin < 3, CFG = struct(); end
    if ~isfield(CFG,'N'),          CFG.N = 30; end
    if ~isfield(CFG,'MaxIt'),      CFG.MaxIt = 30; end
    if ~isfield(CFG,'runs'),       CFG.runs = 10; end
    if ~isfield(CFG,'test_ratio'), CFG.test_ratio = 0.50; end
    if ~isfield(CFG,'seed0'),      CFG.seed0 = 42; end

    % -------------------- locate output dir --------------------
    CORE_DIR = fileparts(mfilename('fullpath'));     % .../run/core
    RUN_DIR  = fileparts(CORE_DIR);                  % .../run
    OUTDIR   = fullfile(RUN_DIR, 'results', DATASET_NAME);
    if ~exist(OUTDIR, 'dir'); mkdir(OUTDIR); end

    % -------------------- load clean data --------------------
    Xall = readmatrix(CLEAN_PATH);
    if isempty(Xall) || size(Xall,2) < 2
        error("CLEAN_PATH invalid or <2 columns: %s", CLEAN_PATH);
    end
    feat = Xall(:, 1:end-1);
    y    = Xall(:, end);

    feat = double(feat);
    y    = double(y);

    % sanity: labels must be 1..K
    u = unique(y);
    if any(isnan(y)) || any(u ~= (1:numel(u))')
        error("Labels must be 1..K with no NaN. Got: %s", mat2str(u'));
    end

    n = size(feat,1);
    d = size(feat,2);
    K = numel(u);

    fprintf("== %s == n=%d d=%d classes=%d\n", DATASET_NAME, n, d, K);

    % -------------------- algorithm list --------------------
    algs = {
        'CSO',       @CSO
        'CSO_KLSR',  @CSO_KLSR
        'JADE',      @JADE
        'JADE_KLSR', @JADE_KLSR
        'GA',        @GA
        'GA_KLSR',   @GA_KLSR
        'PSO',       @PSO
        'PSO_KLSR',  @PSO_KLSR
    };
    A = size(algs,1);

    % containers
    acc_runs = nan(CFG.runs, A);
    err_runs = nan(CFG.runs, A);
    bestX    = cell(CFG.runs, A);

    % bounds for feature selection
    lb = 0; ub = 1; dim = d;

    % -------------------- repeated runs --------------------
    for r = 1:CFG.runs
        rng(CFG.seed0 + r, 'twister');

        % ---- stratified holdout split ----
        [tr_idx, te_idx] = stratified_split(y, CFG.test_ratio);

        Xtr = feat(tr_idx, :);
        ytr = y(tr_idx);
        Xte = feat(te_idx, :);
        yte = y(te_idx);

        % ELM uses shape: (R * Q)
        p_train = Xtr';  t_train = ytr';
        p_test  = Xte';  t_test  = yte';

        fprintf("[run %d/%d] train=%d test=%d\n", r, CFG.runs, numel(ytr), numel(yte));

        % objective: error = 1 - acc
        fobj = @(x) elm_fs_error(x, p_train, t_train, p_test, t_test);

        for a = 1:A
            name = algs{a,1};
            fn   = algs{a,2};

            % run optimizer
            [bestErr, bx, ~] = fn(CFG.N, CFG.MaxIt, lb, ub, dim, fobj);

            % evaluate once more (stable) and store
            bestErr = elm_fs_error(bx, p_train, t_train, p_test, t_test);
            bestAcc = 1 - bestErr;

            err_runs(r,a) = bestErr;
            acc_runs(r,a) = bestAcc;
            bestX{r,a} = bx;

            fprintf("  [%d/%d] %-9s err=%.6f acc=%.6f\n", a, A, name, bestErr, bestAcc);
        end

        % save per-run snapshot
        save(fullfile(OUTDIR, sprintf('run_%02d.mat', r)), ...
            'r','CFG','tr_idx','te_idx','acc_runs','err_runs','bestX');
    end

    % -------------------- summary --------------------
    mean_acc = mean(acc_runs, 1, 'omitnan');
    std_acc  = std(acc_runs,  0, 'omitnan');

    % write summary CSV
    summary_csv = fullfile(OUTDIR, 'summary_accuracy.csv');
    fid = fopen(summary_csv, 'w');
    fprintf(fid, "Algorithm,MeanAcc,StdAcc\n");
    for a = 1:A
        fprintf(fid, "%s,%.10f,%.10f\n", algs{a,1}, mean_acc(a), std_acc(a));
    end
    fclose(fid);

    save(fullfile(OUTDIR, 'summary.mat'), ...
        'CFG','algs','acc_runs','err_runs','bestX','mean_acc','std_acc','CLEAN_PATH');

    fprintf("Saved summary: %s\n", summary_csv);
end

% -------------------- helpers --------------------

function [train_idx, test_idx] = stratified_split(y, test_ratio)
% returns row indices for train/test, stratified by class
    classes = unique(y);
    train_idx = [];
    test_idx  = [];

    for i = 1:numel(classes)
        c = classes(i);
        idx = find(y == c);
        idx = idx(randperm(numel(idx)));

        n_test = round(test_ratio * numel(idx));
        n_test = max(1, min(n_test, numel(idx)-1)); % ensure both sides non-empty

        test_idx  = [test_idx;  idx(1:n_test)]; %#ok<AGROW>
        train_idx = [train_idx; idx(n_test+1:end)]; %#ok<AGROW>
    end

    % shuffle final indices
    test_idx  = test_idx(randperm(numel(test_idx)));
    train_idx = train_idx(randperm(numel(train_idx)));
end

function err = elm_fs_error(x, p_train, t_train, p_test, t_test)
% Feature selection objective using ELM classifier
% - x is continuous in [0,1], we round it to 0/1
% - if no features selected => large penalty
% - else train ELM and compute test error = 1 - accuracy

    id = round(x);
    idx = find(id == 1);

    if isempty(idx)
        err = 1e4;
        return;
    end

    p_tr = p_train(idx, :);
    p_te = p_test(idx, :);

    % ELM settings (same as your FS code)
    num_hiddens = 50;
    TF = 'sig';
    TYPE = 1;

    [IW, B, LW, TF, TYPE] = elmtrain(p_tr, t_train, num_hiddens, TF, TYPE);
    pred = elmpredict(p_te, IW, B, LW, TF, TYPE);

    acc = sum(pred == t_test) / numel(t_test);
    err = 1 - acc;
end
