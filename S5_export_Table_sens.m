function S5_export_Table_sens()
% Export Table for KLSR-CSO sensitivity study
% Input : results_sens_tau_p0/Table_sens.csv  (from S2_summarize_table_sens)
% Output: results_sens_tau_p0/Table_klsr_cso.tex

    inDir = fullfile(pwd, "results_sens_tau_p0");
    inCsv = fullfile(inDir, "Table_sens.csv");
    T = readtable(inCsv);

    % ---- format function name like F01 ----
    FuncStr = arrayfun(@(x) sprintf('F%02d', x), T.Func, 'UniformOutput', false);

    % ---- format optimal (tau,p0) ----
    OptStr = arrayfun(@(a,b) sprintf('(%d, %.2f)', a, b), T.tau_star, T.p0_star, 'UniformOutput', false);

    % ---- number formatting like 8.471E+02 ----
    fmtSci = @(x) format_sci(x);

    baseMed = arrayfun(@(x) fmtSci(x), T.Median_Baseline, 'UniformOutput', false);
    klsrMed = arrayfun(@(x) fmtSci(x), T.Median_KLSRbest, 'UniformOutput', false);

    % ---- improvement (%): keep 1 decimal like CCE ----
    impStr  = arrayfun(@(x) sprintf('%.1f', x), T.ImprovePct, 'UniformOutput', false);

    % ---- time (s): keep 2 decimals like CCE ----
    tBase = arrayfun(@(x) sprintf('%.2f', x), T.MeanTime_Baseline, 'UniformOutput', false);
    tKlsr = arrayfun(@(x) sprintf('%.2f', x), T.MeanTime_KLSRbest, 'UniformOutput', false);

    % ---- write LaTeX (booktabs style like CCE) ----
    outTex = fullfile(inDir, "Table_klsr_cso.tex");
    fid = fopen(outTex, 'w');

    fprintf(fid, "%% Auto-generated from %s\n", inCsv);
    fprintf(fid, "\\begin{table}[t]\n");
    fprintf(fid, "\\centering\n");
    fprintf(fid, "\\caption{Per-function optimal configurations for KLSR--CSO and performance improvements over baseline CSO. Relative improvement is computed as (Baseline -- KLSR)/Baseline $\\times 100\\%%$.}\n");
    fprintf(fid, "\\label{tab:klsr_perf}\n");
    fprintf(fid, "\\setlength{\\tabcolsep}{6pt}\n");
    fprintf(fid, "\\renewcommand{\\arraystretch}{1.05}\n");
    fprintf(fid, "\\begin{tabular}{l c rr c rr}\n");
    fprintf(fid, "\\toprule\n");
    fprintf(fid, "Functions & Optimal $(\\tau,p_0)$ & \\multicolumn{2}{c}{Median Fitness} & Improvement & \\multicolumn{2}{c}{Mean Time (s)}\\\\\n");
    fprintf(fid, "\\cmidrule(lr){3-4} \\cmidrule(lr){6-7}\n");
    fprintf(fid, " &  & Baseline & KLSR & (\\%%) & Baseline & KLSR\\\\\n");
    fprintf(fid, "\\midrule\n");

    for i = 1:height(T)
        % bold KLSR median (like CCE table bolding the improved column)
        fprintf(fid, "%s & %s & %s & \\textbf{%s} & %s & %s & %s \\\\\n", ...
            FuncStr{i}, OptStr{i}, baseMed{i}, klsrMed{i}, impStr{i}, tBase{i}, tKlsr{i});
    end

    fprintf(fid, "\\bottomrule\n");
    fprintf(fid, "\\end{tabular}\n");
    fprintf(fid, "\\end{table}\n");

    fclose(fid);

    fprintf("Saved: %s\n", outTex);
end

function s = format_sci(x)
% Format like 8.471E+02 (CCE style)
    if ~isfinite(x)
        s = "NaN";
        return;
    end
    % use 4 significant digits; adjust if you want 3/5
    s = sprintf('%.4E', x);
    % make exponent have sign and 2 digits like E+02
    % MATLAB already prints E+02, keep it.
end
