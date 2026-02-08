function S2_summarize_table_sens()
    inDir = fullfile(pwd, "results_sens_tau_p0");
    load(fullfile(inDir, "raw_ALL.mat"), "raw");

    nF = numel(raw);
    rows = [];

    for fi = 1:nF
        f = raw(fi).func_id;

        tau_list = raw(fi).tau_list;
        p0_list  = raw(fi).p0_list;

        % ---- baseline stats ----
        base_vals = raw(fi).baseline.best(:);
        base_med  = median(base_vals(~isnan(base_vals) & isfinite(base_vals)));

        base_time_vals = raw(fi).baseline.time(:);
        base_time_mean = mean(base_time_vals(~isnan(base_time_vals) & isfinite(base_time_vals)));

        % ---- scan grid, pick best by median (smaller is better) ----
        best_med = Inf;
        best_it = 1; best_ip = 1;
        best_time_mean = NaN;

        bad_cells = 0;
        total_cells = numel(tau_list) * numel(p0_list);

        for it = 1:numel(tau_list)
            for ip = 1:numel(p0_list)
                vals = squeeze(raw(fi).grid.best(it,ip,:));
                vals = vals(:);

                ok = ~isnan(vals) & isfinite(vals);
                if ~any(ok)
                    bad_cells = bad_cells + 1;
                    continue;
                end

                medv = median(vals(ok));

                if medv < best_med
                    best_med = medv;
                    best_it = it;
                    best_ip = ip;

                    tvals = squeeze(raw(fi).grid.time(it,ip,:));
                    tvals = tvals(:);
                    tok = ~isnan(tvals) & isfinite(tvals);
                    if any(tok)
                        best_time_mean = mean(tvals(tok));
                    else
                        best_time_mean = NaN;
                    end
                end
            end
        end

        tau_best = tau_list(best_it);
        p0_best  = p0_list(best_ip);

        if isempty(base_med) || isnan(base_med) || ~isfinite(base_med)
            improve_pct = NaN;
        else
            improve_pct = (base_med - best_med) / base_med * 100;
        end

        best_vals = squeeze(raw(fi).grid.best(best_it,best_ip,:));
        best_vals = best_vals(:);
        okb = ~isnan(best_vals) & isfinite(best_vals);

        base_ok = ~isnan(base_vals) & isfinite(base_vals);

        base_mean = mean(base_vals(base_ok));
        base_std  = std(base_vals(base_ok));

        best_mean = mean(best_vals(okb));
        best_std  = std(best_vals(okb));

        bad_ratio = bad_cells / total_cells;

        rows = [rows; {f, tau_best, p0_best, ...
            base_med, best_med, improve_pct, ...
            base_mean, base_std, best_mean, best_std, ...
            base_time_mean, best_time_mean, bad_ratio}];

        if bad_cells > 0
            fprintf("F%02d: %d/%d grid cells have all NaN/Inf (bad_ratio=%.2f)\n", ...
                f, bad_cells, total_cells, bad_ratio);
        end
    end

    T = cell2table(rows, 'VariableNames', ...
        {'Func','tau_star','p0_star', ...
         'Median_Baseline','Median_KLSRbest','ImprovePct', ...
         'Mean_Baseline','Std_Baseline','Mean_KLSRbest','Std_KLSRbest', ...
         'MeanTime_Baseline','MeanTime_KLSRbest','BadCellRatio'});

    writetable(T, fullfile(inDir, "Table_sens.csv"));
    save(fullfile(inDir, "Table_sens.mat"), "T");

    disp(T);
    fprintf("Saved: Table_sens.csv\n");
end
