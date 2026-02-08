function svm_sonar_main()
% SVM hyperparam tuning on SONAR (NO CV)
% - Holdout test + internal validation
% - Compare 4 algorithms + their KLSR augmented versions (8 total)
% - N=30, Max_iter=30, runs=10
%
% DATA (from your python inspect):
% - n=207, d=61
% - NO HEADER file
% - features: first 60 cols (numeric)
% - label: last col (categorical 'M'/'R')
%
% NOTE: This dataset MUST be read with ReadVariableNames=false,
%       otherwise first sample row is incorrectly treated as header.

clc;

% -------------------- CONFIG --------------------
N            = 30;
Max_iter     = 30;
runs         = 10;
test_ratio   = 0.50;
val_in_train = 0.25;
seed0        = 42;

DATA_FILE = "/Users/zhengzikun/Desktop/SVM/data/Copy of sonar data.csv"; //change to yours

ALG_DIR = "/Users/zhengzikun/Desktop/SVM/klsr"; //change to yours

OUT_DIR = "/Users/zhengzikun/Desktop/SVM/results"; //change to yours
if ~exist(OUT_DIR, "dir"), mkdir(OUT_DIR); end

% add algorithms
addpath(genpath(ALG_DIR));

% quick existence check
path_check(ALG_DIR);

% algorithms
algs = { ...
    struct("tag","CSO",      "fn",@CSO), ...
    struct("tag","JADE",     "fn",@JADE), ...
    struct("tag","GA",       "fn",@GA), ...
    struct("tag","PSO",      "fn",@PSO), ...
    struct("tag","CSO_KLSR", "fn",@CSO_KLSR), ...
    struct("tag","JADE_KLSR","fn",@JADE_KLSR), ...
    struct("tag","GA_KLSR",  "fn",@GA_KLSR), ...
    struct("tag","PSO_KLSR", "fn",@PSO_KLSR) ...
};

% search space: [log2(C), log2(gamma)]
lb  = [-5, -5];
ub  = [ 5,  5];
dim = 2;

fprintf("\n==================== sonar ====================\n");
fprintf("File: %s\n", DATA_FILE);
fprintf("Setup: N=%d | Max_iter=%d | runs=%d | test=%.2f | val_in_train=%.2f\n\n", ...
    N, Max_iter, runs, test_ratio, val_in_train);

if ~exist(DATA_FILE, "file")
    error("File not found: %s", DATA_FILE);
end

% -------------------- LOAD DATA (STRICT) --------------------
[X_all, y_all] = load_sonar_strict(DATA_FILE);
y_all = double(y_all(:));

fprintf("DBG sonar: n=%d, d=%d, label_unique=%d\n", ...
    size(X_all,1), size(X_all,2), numel(unique(y_all)));

if numel(unique(y_all)) < 2
    error("Label has <2 unique values, cannot train SVM.");
end

holdout_err = zeros(runs, numel(algs));

% -------------------- MAIN RUNS --------------------
for r = 1:runs
    rng(seed0 + r, "twister");

    n = size(X_all, 1);
    idx = randperm(n);
    n_test = round(test_ratio * n);

    test_idx  = idx(1:n_test);
    train_idx = idx(n_test+1:end);

    X_train_full = X_all(train_idx, :);
    y_train_full = y_all(train_idx);
    X_test = X_all(test_idx, :);
    y_test = y_all(test_idx);

    fprintf("[run %d/%d] train=%d test=%d\n", r, runs, numel(train_idx), numel(test_idx));

    % ---- train/val split (NO retry) ----
    ntr = size(X_train_full, 1);
    idx2 = randperm(ntr);

    n_val = max(1, round(val_in_train * ntr));
    val_idx = idx2(1:n_val);
    tr_idx  = idx2(n_val+1:end);

    X_tr  = X_train_full(tr_idx, :);
    y_tr  = y_train_full(tr_idx);
    X_val = X_train_full(val_idx, :);
    y_val = y_train_full(val_idx);

    fobj = @(x) svm_val_error(x, X_tr, y_tr, X_val, y_val);

    % ---- run 8 algorithms ----
    for ai = 1:numel(algs)
        rng(seed0 + r + 1000*ai, "twister");

        A = algs{ai};
        [best_err, best_x] = call_optimizer(A.fn, N, Max_iter, lb, ub, dim, fobj);

        fprintf("  [%d/%d] %s ... best_err=%.6f | best_x=[%.4f %.4f]\n", ...
            ai, numel(algs), A.tag, best_err, best_x(1), best_x(2));

        holdout_err(r, ai) = svm_test_error(best_x, X_train_full, y_train_full, X_test, y_test);
    end
    fprintf("\n");
end

% -------------------- SUMMARY --------------------
fprintf("=== Mean/Std (Holdout Error = 1-Acc) ===\n");
mu = mean(holdout_err, 1);
sd = std(holdout_err, 0, 1);
for ai = 1:numel(algs)
    fprintf("%s: %.6f / %.6f\n", algs{ai}.tag, mu(ai), sd(ai));
end

% save CSV
out_csv = fullfile(OUT_DIR, "sonar_result.csv");
write_result_csv(out_csv, algs, holdout_err);
fprintf("\nSaved: %s\n", out_csv);

end

% ==================== helpers ====================

function path_check(ALG_DIR)
fprintf("[PATH CHECK] Using ALG_DIR: %s\n", ALG_DIR);
names = ["CSO","JADE","GA","PSO","CSO_KLSR","JADE_KLSR","GA_KLSR","PSO_KLSR"];
for i = 1:numel(names)
    f = fullfile(ALG_DIR, names(i) + ".m");
    if exist(f, "file")
        fprintf("  OK: %-10s -> %s\n", names(i), f);
    else
        fprintf("  !! MISSING: %-10s -> %s\n", names(i), f);
    end
end
fprintf("\n");
end

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
    error("Optimizer call failed: %s", ME.message);
end
end

function err = svm_val_error(x, X_tr, y_tr, X_val, y_val)
C = 2.^x(1);
G = 2.^x(2);
sigma = 1/sqrt(2*G);

[Z_tr, mu, sig] = zscore(X_tr);
sig(sig==0) = 1;
Z_val = (X_val - mu) ./ sig;

try
    mdl = fitcsvm(Z_tr, y_tr, ...
        "KernelFunction","rbf", ...
        "KernelScale", sigma, ...
        "BoxConstraint", C, ...
        "Standardize", false);
    yhat = predict(mdl, Z_val);
    err = 1 - mean(double(yhat == y_val));
catch
    err = 1.0;
end
end

function err = svm_test_error(x, X_train, y_train, X_test, y_test)
C = 2.^x(1);
G = 2.^x(2);
sigma = 1/sqrt(2*G);

[Z_tr, mu, sig] = zscore(X_train);
sig(sig==0) = 1;
Z_te = (X_test - mu) ./ sig;

try
    mdl = fitcsvm(Z_tr, y_train, ...
        "KernelFunction","rbf", ...
        "KernelScale", sigma, ...
        "BoxConstraint", C, ...
        "Standardize", false);
    yhat = predict(mdl, Z_te);
    err = 1 - mean(double(yhat == y_test));
catch
    err = 1.0;
end
end

function write_result_csv(out_csv, algs, holdout_err)
tags = strings(1, numel(algs));
for i = 1:numel(algs)
    tags(i) = string(algs{i}.tag);
end

T = array2table(holdout_err, "VariableNames", cellstr(tags));
mu = mean(holdout_err, 1);
sd = std(holdout_err, 0, 1);

T_mean = array2table(mu, "VariableNames", cellstr(tags));
T_std  = array2table(sd, "VariableNames", cellstr(tags));

T2 = [T; T_mean; T_std];
rowNames = ["run_" + string(1:size(T,1)) "mean" "std"];
T2.Properties.RowNames = cellstr(rowNames);

writetable(T2, out_csv, "WriteRowNames", true);
end

function [X, y] = load_sonar_strict(fpath)
% SONAR strict loader (compatible with older MATLAB):
% - use readtable(...,'ReadVariableNames',false) directly
% - expect 61 columns: 60 features + 1 label
% - label is 'M'/'R' -> numeric via grp2idx

% IMPORTANT: do NOT use detectImportOptions here (version differences)
T = readtable(fpath, "ReadVariableNames", false, "Delimiter", ",");

if width(T) ~= 61
    error("Expected 61 columns for sonar, got %d. (Check delimiter / file format)", width(T));
end

% rename to Var1..Var61
T.Properties.VariableNames = "Var" + string(1:width(T));

% label last col
y_raw = T{:, 61};
T(:, 61) = [];

% features
X = table2array(T);
X = double(X);

% fill NaN with median
for k = 1:size(X,2)
    col = X(:,k);
    if any(isnan(col))
        medv = median(col(~isnan(col)));
        if isempty(medv) || isnan(medv), medv = 0; end
        col(isnan(col)) = medv;
        X(:,k) = col;
    end
end

% y to numeric
y_str = strtrim(string(y_raw));
y = double(grp2idx(categorical(y_str)));
y = y(:);

end

