function S9_stats_friedman_compare()
% Friedman test + mean rank for Baseline vs KLSR (within each algorithm)
%
% Input:
%   results_compare/raw_COMPARE.mat  (recommended)
%   OR results_compare/Table_compare.csv (fallback)
%
% Output:
%   results_compare/Stats_Friedman.csv
%
% Notes:
% - We run Friedman across functions (blocks), comparing 2 treatments:
%   Baseline vs KLSR, separately for each algorithm.
% - Metric uses per-function MEDIAN across runs (consistent with your tables).

    inDir = fullfile(pwd, "results_compare");
    rawMat = fullfile(inDir, "raw_COMPARE.mat");
    csvFile = fullfile(inDir, "Table_compare.csv");

    if exist(rawMat, "file")
        S = load(rawMat);
        raw = S.raw;  %#ok<NASGU>
        % Expect raw to be a struct array with fields:
        % raw(i).func_id
        % raw(i).algo_name
        % raw(i).baseline.best (runs)
        % raw(i).klsr.best     (runs)
        useRaw = true;
    elseif exist(csvFile, "file")
        Tcsv = readtable(csvFile);
        useRaw = false;
    else
        error("Cannot find raw_COMPARE.mat or Table_compare_CCE.csv in %s", inDir);
    end

    % --- collect medians by function x algo for baseline/klsr ---
    if useRaw
        % build table-like arrays (support two raw formats):
        % Format A: raw(i).func_id, raw(i).algo_name, raw(i).baseline / raw(i).klsr
        % Format B: raw(i).func_id, raw(i).<ALGO>.baseline / raw(i).<ALGO>.klsr
        funcs = unique([raw.func_id]);
        funcs = funcs(:);

        rows = {};

        if isfield(raw, "algo_name")
            algos = unique(string({raw.algo_name}));
            algos = algos(:);
            for a = 1:numel(algos)
                algo = algos(a);
                for fi = 1:numel(funcs)
                    F = funcs(fi);
                    idx = find([raw.func_id] == F & strcmpi(string({raw.algo_name}), algo), 1);
                    if isempty(idx)
                        continue;
                    end
                    b = raw(idx).baseline.best(:); b = b(isfinite(b) & ~isnan(b));
                    k = raw(idx).klsr.best(:);     k = k(isfinite(k) & ~isnan(k));
                    if isempty(b) || isempty(k)
                        continue;
                    end
                    rows(end+1,:) = {F, char(algo), median(b), median(k)}; %#ok<AGROW>
                end
            end
        else
            % detect algorithm-named subfields (exclude known meta fields)
            meta = ["func_id","dim","N","MaxIt","R","seeds","baseline","klsr","grid","tau_list","p0_list","p1_fixed","nPop"];
            allAlgos = strings(0);
            for i = 1:numel(raw)
                fns = string(fieldnames(raw(i)));
                cand = fns(~ismember(fns, meta));
                allAlgos = union(allAlgos, cand);
            end
            algos = allAlgos(:);

            for a = 1:numel(algos)
                algo = algos(a);
                for fi = 1:numel(funcs)
                    F = funcs(fi);
                    idx = find([raw.func_id] == F, 1);
                    if isempty(idx)
                        continue;
                    end
                    if ~isfield(raw(idx), char(algo))
                        continue;
                    end
                    try
                        b = raw(idx).(char(algo)).baseline.best(:); b = b(isfinite(b) & ~isnan(b));
                        k = raw(idx).(char(algo)).klsr.best(:);     k = k(isfinite(k) & ~isnan(k));
                    catch
                        continue;
                    end
                    if isempty(b) || isempty(k)
                        continue;
                    end
                    rows(end+1,:) = {F, char(algo), median(b), median(k)}; %#ok<AGROW>
                end
            end
        end

        T = cell2table(rows, "VariableNames", ["Func","Algo","Median_Baseline","Median_KLSR"]);
    else
        % your csv already has these columns (judging from your print):
        % Func, Algo, Median_Baseline, Median_KLSR, ...
        % Algo is currently like {'CSO' } as a cell-in-cell → normalize it
        T = Tcsv;
        T.Algo = normalizeAlgoColumn(T.Algo);
    end

    algos = unique(string(T.Algo));
    outRows = {};
    for a = 1:numel(algos)
        algo = algos(a);
        Ta = T(strcmpi(string(T.Algo), algo), :);

        % Ensure each function appears once
        [uF, ia] = unique(Ta.Func, "stable");
        Ta = Ta(ia,:);

        X = [Ta.Median_Baseline, Ta.Median_KLSR]; % nFuncs x 2
        ok = all(isfinite(X),2);
        X = X(ok,:);

        n = size(X,1);
        if n < 3
            warning("%s: too few functions (%d) after filtering. Skipping.", algo, n);
            continue;
        end

        % Friedman: blocks=functions, treatments=2
        % MATLAB friedman expects data matrix nBlocks x nTreat
        [p, tbl, stats] = friedman(X, 1, "off"); %#ok<ASGLU>
        % mean ranks: stats.meanranks is 1 x nTreat
        mr = stats.meanranks;

        outRows(end+1,:) = {char(algo), n, p, mr(1), mr(2)}; %#ok<AGROW>
        fprintf("[%s] Friedman: nFunc=%d, p=%.4g, meanRank(Base)=%.3f, meanRank(KLSR)=%.3f\n", ...
            algo, n, p, mr(1), mr(2));
    end

    Tout = cell2table(outRows, ...
        "VariableNames", ["Algo","nFunc","Friedman_p","MeanRank_Baseline","MeanRank_KLSR"]);

    outCsv = fullfile(inDir, "Stats_Friedman.csv");
    writetable(Tout, outCsv);
    fprintf("Saved: %s\n", outCsv);
end


function Algo = normalizeAlgoColumn(AlgoCol)
% AlgoCol might be:
% - string/cellstr: "CSO"
% - cell of char
% - weird: each entry is a 1x1 cell, like {'CSO'}
    if isstring(AlgoCol)
        Algo = AlgoCol;
        return;
    end
    if iscell(AlgoCol)
        Algo = strings(size(AlgoCol));
        for i = 1:numel(AlgoCol)
            v = AlgoCol{i};
            if iscell(v) && numel(v)==1
                v = v{1};
            end
            if isstring(v), v = char(v); end
            Algo(i) = string(strtrim(v));
        end
        return;
    end
    Algo = string(AlgoCol);
end
