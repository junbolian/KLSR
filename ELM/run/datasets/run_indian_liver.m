function run_indian_liver(CFG)
% Raw file: fs/data/indian_liver_patient.csv
% - Label: Dataset (1/2) keep as 1/2
% - Features: all remaining columns, encode Gender
% - Fill missing: Albumin_and_Globulin_Ratio (mean)
% - Extra: z-score standardization (ELM stability)

    % -------------------- locate paths --------------------
    this_file   = mfilename('fullpath');
    datasets_dir= fileparts(this_file);
    run_dir     = fileparts(datasets_dir);
    root_dir    = fileparts(run_dir);

    RAW = fullfile(root_dir, 'fs', 'data', 'indian_liver_patient.csv');

    OUTDIR = fullfile(run_dir, 'results', 'indian_liver');
    if ~exist(OUTDIR, 'dir'); mkdir(OUTDIR); end
    CLEAN_PATH = fullfile(OUTDIR, 'clean.csv');

    % -------------------- read table --------------------
    T = readtable(RAW, 'VariableNamingRule', 'preserve');

    % Label
    if ~any(strcmpi(T.Properties.VariableNames, 'Dataset'))
        disp(T.Properties.VariableNames);
        error("Missing label column: Dataset");
    end
    y = double(T.Dataset);
    T.Dataset = [];

    if ~all(ismember(unique(y), [1 2]))
        error("Unexpected Dataset label values (expected 1/2).");
    end

    % Encode Gender (Male/Female) -> 1/2
    if any(strcmpi(T.Properties.VariableNames, 'Gender'))
        g = upper(strtrim(string(T.Gender)));
        gi = zeros(size(g));
        gi(g == "MALE") = 1;
        gi(g == "FEMALE") = 2;
        if any(gi == 0)
            bad = unique(g(gi == 0));
            error("Unknown Gender value(s): %s", strjoin(bad, ", "));
        end
        T.Gender = gi;
    end

    % Features to numeric
    X = double(table2array(T));

    % Fill NaNs (mean per column) - handles Albumin_and_Globulin_Ratio missing
    for j = 1:size(X,2)
        cj = X(:, j);
        if any(isnan(cj))
            m = mean(cj(~isnan(cj)));
            if isnan(m), m = 0; end
            cj(isnan(cj)) = m;
            X(:, j) = cj;
        end
    end

    % z-score standardization
    mu = mean(X, 1);
    sig = std(X, 0, 1);
    sig(sig == 0) = 1;
    X = (X - mu) ./ sig;

    writematrix([X, y], CLEAN_PATH);

    fprintf("indian_liver clean saved: %s\n", CLEAN_PATH);
    fprintf("n=%d, d=%d, classes=%d | label_col=Dataset\n", ...
        size(X,1), size(X,2), numel(unique(y)));

    % -------------------- call shared experiment driver --------------------
    if exist('fs_feature_exp', 'file') ~= 2
        error("Missing core driver: fs_feature_exp.m (put it under run/core/)");
    end

    fs_feature_exp(CLEAN_PATH, 'indian_liver', CFG);
end
