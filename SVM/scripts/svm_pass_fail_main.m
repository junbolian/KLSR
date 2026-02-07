function svm_pass_fail_main()
% svm_pass_fail_main.m
% Single-dataset runner: Pass-Fail Data.csv
% - Compare 4 metaheuristics + their KLSR-augmented versions (8 total)
% - NO cross validation
% - Outer holdout: train/test
% - Inner holdout within train: train/val
% - N=30, Max_iter=30, runs=10
%
% Pass-Fail Data (verified by your Python inspector):
%   - columns: student_id, attendance_pct, homework_pct, midterm_score, study_hours_per_week, pass
%   - label: pass (0/1), distribution 1:60, 0:40
%   - drop ID: student_id ONLY (midterm_score is a real feature; do NOT drop)

clc;

% -------------------- CONFIG --------------------
N            = 30;
Max_iter     = 30;
runs         = 10;
test_ratio   = 0.50;
val_in_train = 0.25;
seed0        = 42;

ROOT    = '/Users/zhengzikun/Desktop/SVM';
ALG_DIR = fullfile(ROOT, 'klsr');
fpath   = fullfile(ROOT, 'data', 'Pass-Fail Data.csv');

OUT_DIR = fullfile(ROOT, 'results');
if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end


% -------------------- PATH SETUP --------------------
if ~isfolder(ALG_DIR)
    error('ALG_DIR not found: %s', ALG_DIR);
end
addpath(genpath(ALG_DIR));

% -------------------- OPTIMIZERS --------------------
algs = { ...
    struct('tag','CSO',      'fn_name','CSO'), ...
    struct('tag','JADE',     'fn_name','JADE'), ...
    struct('tag','GA',       'fn_name','GA'), ...
    struct('tag','PSO',      'fn_name','PSO'), ...
    struct('tag','CSO_KLSR', 'fn_name','CSO_KLSR'), ...
    struct('tag','JADE_KLSR','fn_name','JADE_KLSR'), ...
    struct('tag','GA_KLSR',  'fn_name','GA_KLSR'), ...
    struct('tag','PSO_KLSR', 'fn_name','PSO_KLSR') ...
};

% ---- pre-check optimizer functions exist ----
missing = {};
fprintf('\n[PATH CHECK] Using ALG_DIR: %s\n', ALG_DIR);
for i = 1:numel(algs)
    fnm = algs{i}.fn_name;
    w = which(fnm);
    if isempty(w)
        missing{end+1} = fnm; %#ok<AGROW>
    else
        fprintf('  OK: %-10s -> %s\n', fnm, w);
    end
end
if ~isempty(missing)
    fprintf('\n[ERROR] Missing optimizer functions:\n');
    for i = 1:numel(missing)
        fprintf('  - %s\n', missing{i});
    end
    error('Stop: Some optimizer .m files are not found on path.');
end

for i = 1:numel(algs)
    algs{i}.fn = str2func(algs{i}.fn_name);
end

% search space: [log2(C), log2(gamma)]
lb  = [-5, -5];
ub  = [ 5,  5];
dim = 2;

% -------------------- LOAD --------------------
fprintf('\n==================== pass_fail ====================\n');
fprintf('File: %s\n', fpath);
fprintf('Setup: N=%d | Max_iter=%d | runs=%d | test=%.2f | val_in_train=%.2f\n\n', ...
    N, Max_iter, runs, test_ratio, val_in_train);

if ~exist(fpath, 'file')
    error('File not found: %s', fpath);
end

[X_all, y_all] = load_pass_fail_strict(fpath);

fprintf('DBG pass_fail: n=%d, d=%d, label_unique=%d\n', ...
    size(X_all,1), size(X_all,2), numel(unique(y_all)));

if numel(unique(y_all)) < 2
    error('Label has <2 unique values. Something is wrong.');
end

% -------------------- RUNS --------------------
holdout_err = zeros(runs, numel(algs));

for r = 1:runs
    rng(seed0 + r, 'twister');

    n = size(X_all, 1);
    idx = randperm(n);

    n_test = round(test_ratio * n);
    n_test = max(1, min(n-1, n_test));

    test_idx  = idx(1:n_test);
    train_idx = idx(n_test+1:end);

    X_train_full = X_all(train_idx, :);
    y_train_full = y_all(train_idx);
    X_test = X_all(test_idx, :);
    y_test = y_all(test_idx);

    fprintf('[run %d/%d] train=%d test=%d\n', r, runs, numel(train_idx), numel(test_idx));

    % ---- train/val split (NO CV, single split) ----
    ntr = size(X_train_full, 1);
    idx2 = randperm(ntr);

    n_val = round(val_in_train * ntr);
    n_val = max(1, min(ntr-1, n_val));

    val_idx = idx2(1:n_val);
    tr_idx  = idx2(n_val+1:end);

    X_tr  = X_train_full(tr_idx, :);
    y_tr  = y_train_full(tr_idx);
    X_val = X_train_full(val_idx, :);
    y_val = y_train_full(val_idx);

    fobj = @(x) svm_val_error(x, X_tr, y_tr, X_val, y_val);

    % ---- run 8 algorithms ----
    for ai = 1:numel(algs)
        rng(seed0 + r + 1000*ai, 'twister');

        A = algs{ai};
        [best_err, best_x] = call_optimizer(A.fn, N, Max_iter, lb, ub, dim, fobj);

        fprintf('  [%d/%d] %s ... best_err=%.6f | best_x=[%.4f %.4f]\n', ...
            ai, numel(algs), A.tag, best_err, best_x(1), best_x(2));

        holdout_err(r, ai) = svm_test_error(best_x, X_train_full, y_train_full, X_test, y_test);
    end
    fprintf('\n');
end

% -------------------- SUMMARY --------------------
fprintf('=== Mean/Std (Holdout Error = 1-Acc) ===\n');
mu = mean(holdout_err, 1);
sd = std(holdout_err, 0, 1);

for ai = 1:numel(algs)
    fprintf('%s: %.6f / %.6f\n', algs{ai}.tag, mu(ai), sd(ai));
end

out_csv = fullfile(OUT_DIR, 'pass_fail_result.csv');
write_result_csv(out_csv, algs, holdout_err);
fprintf('\nSaved: %s\n', out_csv);

end

% ==================== helpers ====================

function [bestScore, bestX] = call_optimizer(fn, N, Max_iter, lb, ub, dim, fobj)
bestScore = inf;
bestX = zeros(1, dim);

try
    [bestScore, bestX, ~] = fn(N, Max_iter, lb, ub, dim, fobj);
    bestX = bestX(:).';
    return;
catch
end

try
    [bestScore, bestX] = fn(N, Max_iter, lb, ub, dim, fobj);
    bestX = bestX(:).';
    return;
catch ME
    error('Optimizer call failed: %s', ME.message);
end
end

function err = svm_val_error(x, X_tr, y_tr, X_val, y_val)
if numel(unique(y_tr)) < 2
    err = 1.0; return;
end

C = 2.^x(1);
G = 2.^x(2);
sigma = 1 / sqrt(2*G);

[Z_tr, mu, sig] = zscore(X_tr);
sig(sig==0) = 1;
Z_val = (X_val - mu) ./ sig;

try
    mdl = fitcsvm(Z_tr, y_tr, ...
        'KernelFunction','rbf', ...
        'KernelScale', sigma, ...
        'BoxConstraint', C, ...
        'Standardize', false);
    yhat = predict(mdl, Z_val);
    err = 1 - mean(double(yhat == y_val));
catch
    err = 1.0;
end
end

function err = svm_test_error(x, X_train, y_train, X_test, y_test)
if numel(unique(y_train)) < 2
    err = 1.0; return;
end

C = 2.^x(1);
G = 2.^x(2);
sigma = 1 / sqrt(2*G);

[Z_tr, mu, sig] = zscore(X_train);
sig(sig==0) = 1;
Z_te = (X_test - mu) ./ sig;

try
    mdl = fitcsvm(Z_tr, y_train, ...
        'KernelFunction','rbf', ...
        'KernelScale', sigma, ...
        'BoxConstraint', C, ...
        'Standardize', false);
    yhat = predict(mdl, Z_te);
    err = 1 - mean(double(yhat == y_test));
catch
    err = 1.0;
end
end

function write_result_csv(out_csv, algs, holdout_err)
tags = cell(1, numel(algs));
for i = 1:numel(algs)
    tags{i} = algs{i}.tag;
end

T = array2table(holdout_err, 'VariableNames', tags);
mu = mean(holdout_err, 1);
sd = std(holdout_err, 0, 1);

T_mean = array2table(mu, 'VariableNames', tags);
T_std  = array2table(sd, 'VariableNames', tags);

T2 = [T; T_mean; T_std];
rowNames = cell(size(T2,1), 1);
for i = 1:size(T,1), rowNames{i} = sprintf('run_%d', i); end
rowNames{end-1} = 'mean';
rowNames{end}   = 'std';
T2.Properties.RowNames = rowNames;

writetable(T2, out_csv, 'WriteRowNames', true);
end

function [X, y] = load_pass_fail_strict(fpath)
% Loader for Pass-Fail Data.csv
% - label: pass (0/1)
% - drop ID: student_id ONLY
% - keep real features: attendance_pct, homework_pct, midterm_score, study_hours_per_week

T = readtable(fpath, 'PreserveVariableNames', true);

% label
if ~ismember('pass', T.Properties.VariableNames)
    error('Label column "pass" not found. Columns=%s', strjoin(T.Properties.VariableNames, ', '));
end
y = double(T.pass(:));
T.pass = [];

% drop ID
if ismember('student_id', T.Properties.VariableNames)
    T.student_id = [];
end

% predictors numeric
X = table2array(T);
X = double(X);

end
