function S1_run_grid_collect_raw()
% S1_run_grid_collect_raw
% Collect raw results for sensitivity scan of (tau, p0) for KLSR-CSO on CEC2017
% Aligned with main_KLSR_compares:
%   - dim=50, nPop=100
%   - MaxIt = floor(10000*D/nPop)
%   - fobj from Get_Functions_cec2017
%   - baseline: CSO
%   - klsr: CSO_KLSR with kopt (scan tau/p0, fix others)

    clc;

    % ======= Experiment setup (match your main) =======
    nPop     = 100;
    dim      = 50;
    run_times = 10;           % main uses 10
    master_seed = 20250810;

    % Functions: same as your main default (change if you want)
    FuncList  = [1:30];
    skipFuncs = [2];          % keep if you want

    % Sensitivity grid
    tau_grid = [25 50 100];
    p0_grid  = [0.2 0.3 0.5];
    p1_fixed = 0.10;

    % Fixed KLSR params (match your main)
    stall_T        = 7;
    only_on_stall  = true;
    bound_mode     = 'clip';

    fit_opts_default = struct( ...
        'D', 128, ...
        'sigma', 1.0, ...
        'lambda', 1e-3, ...
        'k', min(3*dim, 200), ...
        'df', inf, ...
        'quality_min', 0.6);

    quota_fixed     = 0.10;
    arch_max_fixed  = 500;
    stallG_fixed    = 20;
    alpha_fixed     = [1 0.5 0.25];
    maxeval_fixed   = 1;

    outDir = fullfile(pwd, "results_sens_tau_p0");
    if ~exist(outDir, "dir"), mkdir(outDir); end

    fprintf('============================================================\n');
    fprintf('S1 Collect Raw (Aligned with main)\n');
    fprintf('dim=%d, nPop=%d, runs=%d\n', dim, nPop, run_times);
    fprintf('Scan: tau={%s}, p0={%s}, p1=%.2f\n', ...
        strjoin(string(tau_grid),','), ...
        strjoin(string(p0_grid),','), ...
        p1_fixed);
    fprintf('Functions: [%s]\n', strjoin(string(FuncList),' '));
    fprintf('Output: %s\n', outDir);
    fprintf('============================================================\n\n');

    raw = struct();
    fi_out = 0;

    for F = FuncList
        if ismember(F, skipFuncs)
            fprintf('Skip F%02d\n', F);
            continue;
        end

        % Get CEC function exactly like main
        [lb, ub, dim_true, fobj] = Get_Functions_cec2017(F, dim);
        if isempty(dim_true) || dim_true ~= dim
            dim_true = dim; % safety
        end

        MaxIt = floor((10000 * dim_true) / nPop);

        fprintf('\n========== F%02d (dim=%d, MaxIt=%d) ==========\n', F, dim_true, MaxIt);

        % Fixed seeds per function (same for baseline + all grid)
        rng(master_seed + F);
        seeds = randi(2^31-1, run_times, 1);

        % ---- Baseline: CSO ----
        base_best = zeros(run_times,1);
        base_time = zeros(run_times,1);

        for r = 1:run_times
            rng(seeds(r));
            t0 = tic;
            [bestScore, ~, ~] = CSO(nPop, MaxIt, lb, ub, dim_true, fobj);
            base_time(r) = toc(t0);
            base_best(r) = bestScore;
        end
        fprintf('Baseline median=%.3e, meanTime=%.2fs/run\n', median(base_best), mean(base_time));

        % ---- Grid: KLSR-CSO ----
        A = numel(tau_grid);
        B = numel(p0_grid);

        grid_best = zeros(A, B, run_times);
        grid_time = zeros(A, B, run_times);

        for ia = 1:A
            for ib = 1:B
                kopt = struct( ...
                    'tau', tau_grid(ia), ...
                    'p0',  p0_grid(ib), ...
                    'p1',  p1_fixed, ...
                    'stall_T', stall_T, ...
                    'only_on_stall', only_on_stall, ...
                    'bound_mode', bound_mode, ...
                    'fit', fit_opts_default, ...
                    'quota', quota_fixed, ...
                    'arch_max', arch_max_fixed, ...
                    'stallG', stallG_fixed, ...
                    'alpha', alpha_fixed, ...
                    'max_evals', maxeval_fixed);

                for r = 1:run_times
                    rng(seeds(r));
                    t1 = tic;
                    [bestScore, ~, ~] = CSO_KLSR(nPop, MaxIt, lb, ub, dim_true, fobj, kopt);
                    grid_time(ia,ib,r) = toc(t1);
                    grid_best(ia,ib,r) = bestScore;
                end

                fprintf('Grid tau=%d p0=%.2f median=%.3e time=%.2fs/run\n', ...
                    tau_grid(ia), p0_grid(ib), median(squeeze(grid_best(ia,ib,:))), mean(squeeze(grid_time(ia,ib,:))));
            end
        end

        % ---- Save per-function ----
        fi_out = fi_out + 1;
        raw(fi_out).func_id   = F;
        raw(fi_out).dim       = dim_true;
        raw(fi_out).nPop      = nPop;
        raw(fi_out).MaxIt     = MaxIt;
        raw(fi_out).seeds     = seeds;

        raw(fi_out).baseline.best = base_best;
        raw(fi_out).baseline.time = base_time;

        raw(fi_out).tau_list  = tau_grid;
        raw(fi_out).p0_list   = p0_grid;
        raw(fi_out).p1_fixed  = p1_fixed;
        raw(fi_out).grid.best = grid_best;
        raw(fi_out).grid.time = grid_time;

        save(fullfile(outDir, sprintf("raw_F%02d.mat", F)), "raw", "-v7.3");
    end

    save(fullfile(outDir, "raw_ALL.mat"), "raw", "-v7.3");
    fprintf('\nDone. Raw saved to: %s\n', outDir);
end
