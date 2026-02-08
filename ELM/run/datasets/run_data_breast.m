function run_data_breast(CFG)
% Raw file: fs/data/data_breast.csv
% - Drop: id, Unnamed: 32 (all NaN)
% - Label: diagnosis (M/B) -> 1/2
% - Features: all remaining numeric columns

    % -------------------- locate paths --------------------
    this_file   = mfilename('fullpath');
    datasets_dir= fileparts(this_file);
    run_dir     = fileparts(datasets_dir);
    root_dir    = fileparts(run_dir);

    RAW = fullfile(root_dir, 'fs', 'data', 'data_breast.csv');

    OUTDIR = fullfile(run_dir, 'results', 'data_breast');
    if ~exist(OUTDIR, 'dir'); mkdir(OUTDIR); end
    CLEAN_PATH = fullfile(OUTDIR, 'clean.csv');

    % -------------------- force correct headers --------------------
    % Read header line explicitly (first row)
    hdr = readcell(RAW, 'Range', '1:1');
    hdr = string(hdr);
    hdr = strtrim(hdr);

    % Now read the rest as data (from row 2), WITHOUT variable names
    T = readtable(RAW, 'ReadVariableNames', false, 'HeaderLines', 1);

    % Assign correct names
    if numel(hdr) ~= width(T)
        error("Header count (%d) != data columns (%d).", numel(hdr), width(T));
    end
    T.Properties.VariableNames = matlab.lang.makeUniqueStrings(matlab.lang.makeValidName(hdr));

    % -------------------- drop id + unnamed (if present) --------------------
    vn = string(T.Properties.VariableNames);
    vn_norm = lower(regexprep(strtrim(vn), "\s+", ""));

    id_idx = find(vn_norm == "id", 1);
    if ~isempty(id_idx)
        T(:, id_idx) = [];
        vn = string(T.Properties.VariableNames);
        vn_norm = lower(regexprep(strtrim(vn), "\s+", ""));
    end

    % Drop 'Unnamed: 32' by name OR by all-NaN
    unnamed_idx = find(contains(lower(vn), "unnamed"), 1);
    if ~isempty(unnamed_idx)
        T(:, unnamed_idx) = [];
        vn = string(T.Properties.VariableNames);
        vn_norm = lower(regexprep(strtrim(vn), "\s+", ""));
    else
        drop_cols = false(1, width(T));
        for j = 1:width(T)
            col = T{:, j};
            if isnumeric(col)
                drop_cols(j) = all(isnan(double(col)));
            end
        end
        if any(drop_cols)
            T(:, drop_cols) = [];
            vn = string(T.Properties.VariableNames);
            vn_norm = lower(regexprep(strtrim(vn), "\s+", ""));
        end
    end

    % -------------------- label: diagnosis --------------------
    label_idx = find(vn_norm == "diagnosis", 1);
    if isempty(label_idx)
        % show names for debugging
        disp(T.Properties.VariableNames);
        error("Missing label column: diagnosis");
    end

    y_raw = string(T{:, label_idx});
    T(:, label_idx) = [];

    y_raw = upper(strtrim(y_raw));
    y = zeros(size(y_raw));
    y(y_raw == "M") = 1;
    y(y_raw == "B") = 2;
    if any(y == 0)
        bad = unique(y_raw(y == 0));
        error("Unknown diagnosis label(s): %s", strjoin(bad, ", "));
    end

    % -------------------- features -> numeric --------------------
    X = double(table2array(T)); % now should be purely numeric
    if any(isnan(X(:)))
        % fill NaNs just in case
        for j = 1:size(X,2)
            cj = X(:, j);
            m = mean(cj(~isnan(cj)));
            if isnan(m), m = 0; end
            cj(isnan(cj)) = m;
            X(:, j) = cj;
        end
    end

    writematrix([X, y], CLEAN_PATH);

    fprintf("data_breast clean saved: %s\n", CLEAN_PATH);
    fprintf("n=%d, d=%d, classes=%d\n", size(X,1), size(X,2), numel(unique(y)));

    % -------------------- call shared experiment driver --------------------
    if exist('fs_feature_exp', 'file') ~= 2
        error("Missing core driver: fs_feature_exp.m (put it under run/core/)");
    end

    fs_feature_exp(CLEAN_PATH, 'data_breast', CFG);
end
