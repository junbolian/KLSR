function run_data_csv(CFG)
% Raw file: fs/data/data.csv
% - Drop: ID
% - Label: class (P/H) -> 1/2
% - Features: all remaining numeric columns

    % -------------------- locate paths --------------------
    this_file   = mfilename('fullpath');
    datasets_dir= fileparts(this_file);
    run_dir     = fileparts(datasets_dir);
    root_dir    = fileparts(run_dir);

    RAW = fullfile(root_dir, 'fs', 'data', 'data.csv');

    OUTDIR = fullfile(run_dir, 'results', 'data_csv');
    if ~exist(OUTDIR, 'dir'); mkdir(OUTDIR); end
    CLEAN_PATH = fullfile(OUTDIR, 'clean.csv');

    % -------------------- read table --------------------
    T = readtable(RAW);

    % Drop ID column
    if any(strcmpi(T.Properties.VariableNames, 'ID'))
        T.ID = [];
    else
        % fallback: first column if it is non-numeric
        if ~isnumeric(T{:,1})
            T(:,1) = [];
        end
    end

    % Label: class
    if ~any(strcmpi(T.Properties.VariableNames, 'class'))
        error("Missing label column: class");
    end
    y_raw = string(T.class);
    T.class = [];

    % Encode label: P/H -> 1/2
    y_raw = upper(strtrim(y_raw));
    y = zeros(size(y_raw));
    y(y_raw == "P") = 1;
    y(y_raw == "H") = 2;

    if any(y == 0)
        bad = unique(y_raw(y == 0));
        error("Unknown class label(s): %s", strjoin(bad, ", "));
    end

    % Features
    X = double(table2array(T));

    writematrix([X, y], CLEAN_PATH);

    fprintf("data_csv clean saved: %s\n", CLEAN_PATH);
    fprintf("n=%d, d=%d, classes=%d\n", size(X,1), size(X,2), numel(unique(y)));

    % -------------------- call shared experiment driver --------------------
    if exist('fs_feature_exp', 'file') ~= 2
        error("Missing core driver: fs_feature_exp.m (put it under run/core/)");
    end

    fs_feature_exp(CLEAN_PATH, 'data_csv', CFG);
end
