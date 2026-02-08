function S6_run_compare_collect_raw()
% Comparative experiments (Section 4):
% Run Baseline vs KLSR for 4 optimizers: CSO / PSO / GA / JADE
% Across CEC2017 (exclude F02), D=50, N=100, R=10
%
% Output:
%   results_compare/raw_COMPARE.mat
%
% Requirements in path:
%   - Get_Functions_cec2017(F, dim)
%   - Baselines: CSO, PSO, GA, JADE
%   - KLSR versions: CSO_KLSR, PSO_KLSR, GA_KLSR, JADE_KLSR
%
% Notes:
%   - If your GA/JADE function names differ, edit resolve_solver_name() below.

    clc; clear; close all;

    outDir = fullfile(pwd, "results_compare");
    if ~exist(outDir,"dir"), mkdir(outDir); end

    % ---- experiment constants ----
    dim  = 50;
    N    = 100;
    R    = 10;

    funcs = 1:30;
    funcs(funcs==2) = []; % exclude F02

    MaxFEs = 10000 * dim;
    MaxIt  = floor(MaxFEs / N);

    % ---- KLSR fixed params (from your sensitivity) ----
    kopt = struct();
    kopt.tau = 25;
    kopt.p0  = 0.30;
    kopt.p1  = 0.10;

    % (optional) pass-through extra fields if your KLSR uses them
    % kopt.enable_klsr = true;  % if needed by your implementation

    % ---- algorithms to compare ----
    algoList = {"CSO","PSO","GA","JADE"};

    % ---- seeds (identical across baseline vs klsr) ----
    master_seed = 20260127;
    rng(master_seed);

    seeds = randi(2^31-1, R, 1);

    raw = struct([]);
    idxF = 0;

    fprintf("============================================================\n");
    fprintf("Comparisons: D=%d, N=%d, MaxFEs=%d, MaxIt=%d, R=%d\n", dim, N, MaxFEs, MaxIt, R);
    fprintf("Functions: 29 (exclude F02)\n");
    fprintf("KLSR fixed: tau=%d, p0=%.2f, p1=%.2f\n", kopt.tau, kopt.p0, kopt.p1);
    fprintf("Algorithms: ");
        for i = 1:numel(algoList)
            fprintf("%s ", algoList{i});
        end
    fprintf("\n");
    fprintf("Output dir: %s\n", outDir);
    fprintf("============================================================\n\n");

    for F = funcs
        idxF = idxF + 1;

        [lb, ub, dim_true, fobj] = Get_Functions_cec2017(F, dim);
        if isempty(dim_true) || dim_true ~= dim
            dim_true = dim;
        end

        fprintf("=== F%02d | dim=%d | MaxIt=%d ===\n", F, dim_true, MaxIt);

        raw(idxF).func_id = F;
        raw(idxF).dim     = dim_true;
        raw(idxF).N       = N;
        raw(idxF).MaxIt   = MaxIt;
        raw(idxF).R       = R;
        raw(idxF).seeds   = seeds;

        for a = 1:numel(algoList)
            algo = algoList{a};

            % ---- baseline runs ----
            base_best = nan(R,1);
            base_time = nan(R,1);

            for r = 1:R
                [base_best(r), base_time(r)] = run_one_algo(algo, false, F, seeds(r), N, MaxIt, lb, ub, dim_true, fobj, kopt, outDir);
            end

            % ---- KLSR runs ----
            klsr_best = nan(R,1);
            klsr_time = nan(R,1);

            for r = 1:R
                [klsr_best(r), klsr_time(r)] = run_one_algo(algo, true, F, seeds(r), N, MaxIt, lb, ub, dim_true, fobj, kopt, outDir);
            end

            raw(idxF).(algo).baseline.best = base_best;
            raw(idxF).(algo).baseline.time = base_time;

            raw(idxF).(algo).klsr.best = klsr_best;
            raw(idxF).(algo).klsr.time = klsr_time;

            % quick console summary (median)
            bm = median(base_best(isfinite(base_best)));
            km = median(klsr_best(isfinite(klsr_best)));
            imp = (bm - km) / bm * 100;

            fprintf("  %-4s | median(b)=%.3e  median(k)=%.3e  imp=%.2f%%\n", algo, bm, km, imp);
        end

        fprintf("\n");
    end

    save(fullfile(outDir, "raw_COMPARE.mat"), "raw");

    fprintf("Done. Raw saved to: %s\n", outDir);
end


function [best_final, time_sec] = run_one_algo(algo, use_klsr, func_id, seed, N, MaxIt, lb, ub, dim, fobj, kopt, outDir)
% Robust single-run wrapper: always returns finite outputs (or Inf on failure)

    best_final = Inf;
    time_sec   = NaN;

    rng(seed, 'twister');
    t0 = tic;

    try
        % resolve solver function name
        if use_klsr
            solver = resolve_solver_name(algo, true);
        else
            solver = resolve_solver_name(algo, false);
        end

        % ---- call patterns (try a few common signatures) ----
        % Pattern A: (N, MaxIt, lb, ub, dim, fobj)
        % Pattern B: (N, MaxIt, lb, ub, dim, fobj, kopt)
        % Pattern C: some codes accept scalar lb/ub; but we pass what Get_Functions returns.

        if use_klsr
            try
                [bestScore, ~, ~] = feval(solver, N, MaxIt, lb, ub, dim, fobj, kopt);
            catch
                % fallback: sometimes last arg is params struct with tau/p0/p1 fields directly
                [bestScore, ~, ~] = feval(solver, N, MaxIt, lb, ub, dim, fobj, struct("tau",kopt.tau,"p0",kopt.p0,"p1",kopt.p1));
            end
        else
            [bestScore, ~, ~] = feval(solver, N, MaxIt, lb, ub, dim, fobj);
        end

        best_final = bestScore;
        time_sec = toc(t0);

        if ~isfinite(best_final)
            error("run_one_algo:BestNotFinite", "best_final not finite.");
        end

    catch ME
        time_sec = toc(t0);
        best_final = Inf;

        % log
        log_path = fullfile(outDir, "error_log_compare.txt");
        fid = fopen(log_path, "a");
        if fid ~= -1
            fprintf(fid, "=== %s | F%02d | algo=%s | klsr=%d | seed=%d ===\n", datestr(now,31), func_id, algo, use_klsr, seed);
            fprintf(fid, "ERROR: %s\n", ME.message);
            for k = 1:numel(ME.stack)
                fprintf(fid, "  at %s (line %d)\n", ME.stack(k).name, ME.stack(k).line);
            end
            fprintf(fid, "\n");
            fclose(fid);
        end
    end
end


function solver = resolve_solver_name(algo, use_klsr)
% Adjust here if your function names differ.
% MUST exist on MATLAB path.

    algo = upper(string(algo));

    if use_klsr
        switch algo
            case "CSO"
                solver = "CSO_KLSR";
            case "PSO"
                solver = "PSO_KLSR";
            case "GA"
                solver = "GA_KLSR";
            case "JADE"
                solver = "JADE_KLSR";
            otherwise
                error("Unknown algo: %s", algo);
        end
    else
        switch algo
            case "CSO"
                solver = "CSO";
            case "PSO"
                solver = "PSO";
            case "GA"
                solver = "GA";
            case "JADE"
                solver = "JADE";
            otherwise
                error("Unknown algo: %s", algo);
        end
    end
end
