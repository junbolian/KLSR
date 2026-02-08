function run_sonar(CFG)
% Run SONAR dataset (one-file version):
% 1) Clean raw CSV -> run/results/sonar/clean.csv
% 2) Call shared experiment driver: fs_feature_exp(clean.csv, 'sonar', CFG)
%
% Assumptions:
% - Raw file is CSV with 60 numeric features + 1 label column (R/M)
% - Label is the last column

    % -------------------- locate paths --------------------
    RUN_DIR  = fileparts(mfilename('fullpath')); % .../run/datasets
    RUN_DIR  = fileparts(RUN_DIR);               % .../run
    ROOT_DIR = fileparts(RUN_DIR);               % .../FeatureSelectionKLSR

    % >>> IMPORTANT: set the raw filename here to match your fs/data <<<
    RAW = fullfile(ROOT_DIR, 'fs', 'data', 'Copy of sonar data.csv');

    OUTDIR = fullfile(RUN_DIR, 'results', 'sonar');
    if ~exist(OUTDIR, 'dir'); mkdir(OUTDIR); end
    CLEAN_PATH = fullfile(OUTDIR, 'clean.csv');

    % -------------------- clean step (inlined) --------------------
    % Read as table without variable names, because many sonar files have a numeric first row header.
    T = readtable(RAW, 'ReadVariableNames', false);

    if width(T) < 2
        error("SONAR raw file must have >=2 columns. Got width=%d", width(T));
    end

    % Last column is label (R/M), others are features
    X = T{:, 1:end-1};
    y_raw = T{:, end};

    % Force numeric features
    X = double(X);

    % Encode label: R->1, M->2 (fits FS requirement: label=1..K)
    y_raw = string(y_raw);
    y = zeros(size(y_raw));
    y(y_raw == "R") = 1;
    y(y_raw == "M") = 2;

    % Basic sanity checks
    if any(y == 0)
        bad = unique(y_raw(y == 0));
        error("Unknown label(s) found in sonar: %s", strjoin(bad, ", "));
    end

    writematrix([X, y], CLEAN_PATH);

    fprintf("Sonar clean saved: %s\n", CLEAN_PATH);
    fprintf("n=%d, d=%d, classes=%d\n", size(X,1), size(X,2), numel(unique(y)));

    % -------------------- call shared experiment driver --------------------
    if exist('fs_feature_exp', 'file') ~= 2
        error("Missing core driver: fs_feature_exp.m (put it under run/core/)");
    end

    fs_feature_exp(CLEAN_PATH, 'sonar', CFG);
end
