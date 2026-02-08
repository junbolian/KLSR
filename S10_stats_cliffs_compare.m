function S10_stats_cliffs_compare()
% Cliff's delta effect size for Baseline vs KLSR (within each algorithm)
%
% Input:
%   results_compare/raw_COMPARE.mat (recommended)
%   (If you only have Table_compare.csv, it is NOT enough for Cliff's delta
%    because delta needs per-run samples, not just medians.)
%
% Output:
%   results_compare/Stats_Cliffs_perFunc.csv
%   results_compare/Stats_Cliffs_summary.csv
%
% Cliff's delta:
%   δ = P(X > Y) - P(X < Y), for minimization we compare Baseline vs KLSR.
%   Here we compute δ for Baseline minus KLSR using raw per-run best fitness.

    inDir = fullfile(pwd, "results_compare");
    rawMat = fullfile(inDir, "raw_COMPARE.mat");
    if ~exist(rawMat, "file")
        error("Need raw_COMPARE.mat (per-run data) for Cliff's delta. Not found: %s", rawMat);
    end
    S = load(rawMat);
    raw = S.raw;

    funcs = unique([raw.func_id]);
    funcs = funcs(:);

    % Support two raw formats:
    % A) raw(i).algo_name + raw(i).baseline / raw(i).klsr
    % B) raw(i).<ALGO>.baseline / raw(i).<ALGO>.klsr
    if isfield(raw, "algo_name")
        algos = unique(string({raw.algo_name}));
        algos = algos(:);
        formatA = true;
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
        formatA = false;
    end

    perFuncRows = {};
    summaryRows = {};

    for a = 1:numel(algos)
        algo = algos(a);

        deltas = [];
        npos = 0; nneg = 0; nzero = 0;
        nvalid = 0;

        for fi = 1:numel(funcs)
            F = funcs(fi);

            if formatA
                idx = find([raw.func_id] == F & strcmpi(string({raw.algo_name}), algo), 1);
                if isempty(idx)
                    continue;
                end
                b = raw(idx).baseline.best(:);
                k = raw(idx).klsr.best(:);
            else
                idx = find([raw.func_id] == F, 1);
                if isempty(idx)
                    continue;
                end
                if ~isfield(raw(idx), char(algo))
                    continue;
                end
                try
                    b = raw(idx).(char(algo)).baseline.best(:);
                    k = raw(idx).(char(algo)).klsr.best(:);
                catch
                    continue;
                end
            end

            b = b(isfinite(b) & ~isnan(b));
            k = k(isfinite(k) & ~isnan(k));

            if numel(b) < 2 || numel(k) < 2
                continue;
            end

            % Cliff's delta: baseline vs klsr
            % For minimization, "improvement" means KLSR smaller than baseline.
            % δ as defined (b,k): positive δ means baseline tends to be larger than klsr → good for KLSR.
            d = cliffs_delta(b, k);

            nvalid = nvalid + 1;
            deltas(end+1,1) = d; %#ok<AGROW>

            if d > 0
                npos = npos + 1;
            elseif d < 0
                nneg = nneg + 1;
            else
                nzero = nzero + 1;
            end

            mag = cliffs_magnitude(abs(d));

            perFuncRows(end+1,:) = {F, char(algo), d, mag, numel(b), numel(k)}; %#ok<AGROW>
        end

        if nvalid == 0
            warning("[%s] No valid functions for Cliff's delta.", algo);
            continue;
        end

        medD = median(deltas);
        meanD = mean(deltas);

        summaryRows(end+1,:) = {char(algo), nvalid, ...
            npos, nzero, nneg, ...
            medD, meanD, cliffs_magnitude(abs(medD))}; %#ok<AGROW>

        fprintf("[%s] Cliff's delta: valid=%d, +/0/- = %d/%d/%d, median=%.3f, mean=%.3f\n", ...
            algo, nvalid, npos, nzero, nneg, medD, meanD);
    end

    Tper = cell2table(perFuncRows, ...
        "VariableNames", ["Func","Algo","CliffsDelta","Magnitude","nRun_Baseline","nRun_KLSR"]);
    Tout = cell2table(summaryRows, ...
        "VariableNames", ["Algo","nFunc_valid","+","0","-","Delta_median","Delta_mean","Mag_median"]);

    out1 = fullfile(inDir, "Stats_Cliffs_perFunc.csv");
    out2 = fullfile(inDir, "Stats_Cliffs_summary.csv");
    writetable(Tper, out1);
    writetable(Tout, out2);

    fprintf("Saved: %s\n", out1);
    fprintf("Saved: %s\n", out2);
end


function d = cliffs_delta(x, y)
% Compute Cliff's delta δ = P(x>y) - P(x<y)
% x,y: vectors (baseline, klsr)
    x = x(:); y = y(:);
    nx = numel(x); ny = numel(y);

    % O(nx*ny) is fine for your case (R=10)
    gt = 0; lt = 0;
    for i = 1:nx
        gt = gt + sum(x(i) > y);
        lt = lt + sum(x(i) < y);
    end
    d = (gt - lt) / (nx * ny);
end


function mag = cliffs_magnitude(ad)
% Magnitude thresholds (commonly used):
% negligible < 0.147
% small      < 0.33
% medium     < 0.474
% large      >= 0.474
    if ad < 0.147
        mag = "negligible";
    elseif ad < 0.33
        mag = "small";
    elseif ad < 0.474
        mag = "medium";
    else
        mag = "large";
    end
end
