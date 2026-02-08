function run_alzheimers(CFG)
% Run Alzheimer's dataset (one-file version):
% 1) Clean raw CSV -> run/results/alzheimers/clean.csv
% 2) Call shared experiment driver: fs_feature_exp(clean.csv, 'alzheimers', CFG)

    % -------------------- locate paths --------------------
    RUN_DIR  = fileparts(mfilename('fullpath')); 
    RUN_DIR  = fileparts(RUN_DIR);               
    ROOT_DIR = fileparts(RUN_DIR);               

    RAW = fullfile(ROOT_DIR, 'fs', 'data', 'alzheimers_disease_data.csv');

    OUTDIR = fullfile(RUN_DIR, 'results', 'alzheimers');
    if ~exist(OUTDIR, 'dir'); mkdir(OUTDIR); end
    CLEAN_PATH = fullfile(OUTDIR, 'clean.csv');

    % -------------------- clean step (inlined) --------------------
    T = readtable(RAW);

    % Drop non-feature columns
    if any(strcmpi(T.Properties.VariableNames, 'DoctorInCharge'))
        T.DoctorInCharge = [];
    end
    if any(strcmpi(T.Properties.VariableNames, 'PatientID'))
        T.PatientID = [];
    end

    % Label column
    if ~any(strcmpi(T.Properties.VariableNames, 'Diagnosis'))
        error("Missing label column: Diagnosis");
    end
    y = double(T.Diagnosis);
    T.Diagnosis = [];

    % Numeric feature matrix
    X = double(table2array(T));

    if all(ismember(unique(y), [0 1]))
        y = y + 1;
    else
        u = unique(y);
        y2 = zeros(size(y));
        for i = 1:numel(u)
            y2(y == u(i)) = i;
        end
        y = y2;
    end

    writematrix([X, y], CLEAN_PATH);

    fprintf("Alzheimers clean saved: %s\n", CLEAN_PATH);
    fprintf("n=%d, d=%d, classes=%d\n", size(X,1), size(X,2), numel(unique(y)));

    % -------------------- call shared experiment driver --------------------
    if exist('fs_feature_exp', 'file') ~= 2
        error("Missing core driver: fs_feature_exp.m (put it under run/core/)");
    end

    fs_feature_exp(CLEAN_PATH, 'alzheimers', CFG);
end
