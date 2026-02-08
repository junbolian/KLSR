function run_m_of_n(CFG)
% Run M-of-n dataset (one-file version):
% 1) Clean raw CSV -> run/results/m_of_n/clean.csv
% 2) Call shared experiment driver: fs_feature_exp(clean.csv, 'm_of_n', CFG)
%
% Raw file: fs/data/M-of-n.csv
% - numeric features
% - label already is 1/2 (1..K), usually last column

    % -------------------- locate paths --------------------
    RUN_DIR  = fileparts(mfilename('fullpath')); % .../run/datasets
    RUN_DIR  = fileparts(RUN_DIR);               % .../run
    ROOT_DIR = fileparts(RUN_DIR);               % .../FeatureSelectionKLSR

    RAW = fullfile(ROOT_DIR, 'fs', 'data', 'M-of-n.csv');

    OUTDIR = fullfile(RUN_DIR, 'results', 'm_of_n');
    if ~exist(OUTDIR, 'dir'); mkdir(OUTDIR); end
    CLEAN_PATH = fullfile(OUTDIR, 'clean.csv');

    % -------------------- clean step (inlined) --------------------
    Xall = readmatrix(RAW);
    if isempty(Xall) || size(Xall,2) < 2
        error("M-of-n.csv invalid or <2 columns: %s", RAW);
    end
    Xall = double(Xall);

    feat = Xall(:, 1:end-1);
    y = Xall(:, end);

    % Ensure label is 1..K (should already be)
    u = unique(y);
    if all(ismember(u, [0 1]))
        y = y + 1; % just in case
    else
        % if not 1..K, remap
        u = unique(y);
        if ~isequal(sort(u(:))', 1:numel(u))
            y2 = zeros(size(y));
            for i = 1:numel(u)
                y2(y == u(i)) = i;
            end
            y = y2;
        end
    end

    writematrix([feat, y], CLEAN_PATH);

    fprintf("M-of-n clean saved: %s\n", CLEAN_PATH);
    fprintf("n=%d, d=%d, classes=%d\n", size(feat,1), size(feat,2), numel(unique(y)));

    % -------------------- call shared experiment driver --------------------
    if exist('fs_feature_exp', 'file') ~= 2
        error("Missing core driver: fs_feature_exp.m (put it under run/core/)");
    end

    fs_feature_exp(CLEAN_PATH, 'm_of_n', CFG);
end
