function make_cce_paper_tables(matfile)
% ============================================================
% Author: Junbo Jacob Lian
% Post-processing for CCE paper figures & tables (Nature-style)
% - Baseline row explicitly shown in rank plots (rho=0)
% - Each bar top annotated with two-decimal average rank
% - Export PNG (Rank.png alias), CSV, LaTeX tables
% - Export a comprehensive Excel with ALL table data needed:
%     * AvgRank matrix + axes (numeric & labels)
%     * Per-function best summary (tau,rho, medians, improvement%%, times, speedup)
%     * Full medGrid & timeGrid for every function + baseline stats
% Usage:
%   make_cce_paper_tables('results_CCE_PSO_CEC2017_D100_compare.mat')
% ============================================================

clc; close all;

% ---------- Load and sanity ----------
S = load(matfile);
All       = S.All;
tau_grid  = S.tau_grid;
rho_grid  = S.rho_grid;

if isfield(S,'FuncList'),   FuncList = S.FuncList;   else, FuncList = 1:numel(All); end
if isfield(S,'skipFuncs'),  skipFuncs = S.skipFuncs; else, skipFuncs = [];          end
if isfield(S,'algoMode'),   algoMode = S.algoMode;   else, algoMode = 'ALGO';       end

tauAxis = tau_grid(:)';           % 1×A
rhoAxis = [0, rho_grid(:)'];      % 1×(B+1)  (rho=0 => Baseline row)
A = numel(tauAxis);
B = numel(rhoAxis) - 1;

% ---------- Average rank across functions ----------
[avgRank, cntF] = compute_avg_ranks(All, FuncList, skipFuncs, tauAxis, rhoAxis);
if cntF == 0
    error('No valid functions found to compute avg rank.');
end

% Best config by avg rank
[bestVal, bestIdx] = min(avgRank(:));
[iRho, jTau] = ind2sub(size(avgRank), bestIdx);
best_rho = rhoAxis(iRho);
best_tau = tauAxis(jTau);

% ---------- Draw rank 3D bar & surface ----------
[figBar, figSurf] = draw_avg_rank_plots(avgRank, tauAxis, rhoAxis, algoMode, cntF, best_tau, best_rho);

% ---------- Save figures ----------
[pth, nam] = fileparts(matfile);
if isempty(pth), pth = pwd; end
pngBar   = fullfile(pth, [nam '-AvgRank3D.png']);
pngSurf  = fullfile(pth, [nam '-AvgRankSurf.png']);
pngAlias = fullfile(pth, 'Rank.png');   % paper alias

save_png(figBar, pngBar, 500);
save_png(figSurf, pngSurf, 500);
try, copyfile(pngBar, pngAlias); end

% ---------- Save AvgRank CSV & LaTeX ----------
csvRank = fullfile(pth, [nam '-AvgRank3D.csv']);
texRank = fullfile(pth, [nam '-AvgRank3D.tex']);
write_avg_rank_csv(csvRank, avgRank, tauAxis, rhoAxis);
write_avg_rank_tex(texRank, avgRank, tauAxis, rhoAxis, best_tau, best_rho, cntF);

% ---------- Build per-function "grid-best" summary ----------
SumT = build_per_function_summary(All, FuncList, skipFuncs, tauAxis, rhoAxis);

% Save per-function CSV & LaTeX (longtable)
csvBest = fullfile(pth, [nam '-PerFuncBest.csv']);
writetable(SumT, csvBest);

texBest = fullfile(pth, [nam '-PerFuncBest.tex']);
write_perfunc_longtable(texBest, SumT);

% ---------- Export comprehensive Excel ----------
xlsxFile = fullfile(pth, ['PaperData_' nam '.xlsx']);
export_excel_all(xlsxFile, All, FuncList, skipFuncs, tauAxis, rhoAxis, avgRank, SumT);

fprintf('\nSaved:\n  3D bar   : %s\n  3D surf  : %s\n  (alias)  : %s\n  CSV rank : %s\n  TEX rank : %s\n  CSV best : %s\n  TEX best : %s\n  Excel    : %s\n', ...
    pngBar, pngSurf, pngAlias, csvRank, texRank, csvBest, texBest, xlsxFile);
fprintf('Best average rank at tau=%d, rho=%.2f (value=%.2f)\n', best_tau, best_rho, bestVal);

end

% ============================ helpers ============================

function [avgRank, cntF] = compute_avg_ranks(All, FuncList, skipFuncs, tauAxis, rhoAxis)
A = numel(tauAxis);
B = numel(rhoAxis) - 1;
sumRanks = zeros(B+1, A);
cntF = 0;

for F = FuncList
    if ismember(F, skipFuncs), continue; end
    if F>numel(All) || isempty(All(F)) || ~isfield(All(F),'grid') || ~isfield(All(F).grid,'medGrid')
        continue;
    end
    medGrid = All(F).grid.medGrid;  % A×B expected
    if isempty(medGrid), continue; end
    if ~isequal(size(medGrid), [A, B])
        if isequal(size(medGrid), [B, A]), medGrid = medGrid.'; else, continue; end
    end
    baseMed = All(F).baseline.med;
    if isempty(baseMed) || ~isscalar(baseMed) || ~isfinite(baseMed), continue; end

    ranksMat_F = nan(B+1, A);
    for j=1:A
        col_vals = [baseMed ; medGrid(j,:)']; % (B+1)×1
        % smaller is better
        if exist('tiedrank','file')
            rcol = tiedrank(col_vals);
        else
            [~,ord] = sort(col_vals,'ascend');
            rcol = zeros(size(col_vals)); rcol(ord)=1:numel(col_vals);
        end
        ranksMat_F(:, j) = rcol;
    end
    sumRanks = sumRanks + ranksMat_F;
    cntF = cntF + 1;
end

avgRank = sumRanks / max(1,cntF);
end

function [fig1, fig2] = draw_avg_rank_plots(avgRank, tauAxis, rhoAxis, algoMode, cntF, best_tau, best_rho)
set(0,'DefaultAxesFontName','Helvetica','DefaultTextFontName','Helvetica');

% Gentle, print-friendly colormap
cm = parula(256);

% ---- 3D bar ----
fig1 = figure('Color','w','Position',[220 80 980 640]);
bh = bar3(avgRank, 'detached'); %#ok<NASGU>
colormap(cm);
caxis([min(avgRank(:)) max(avgRank(:))]);
cb = colorbar; cb.Label.String = 'Average rank (lower is better)';

ax = gca; grid on; box on; ax.LineWidth=0.8; ax.FontSize=11;
A = numel(tauAxis);
B1 = numel(rhoAxis);

% Labels: Baseline + rho values
yTickLbl = [{'Baseline'}, arrayfun(@(y)sprintf('%.2f',y), rhoAxis(2:end),'uni',0)];
set(ax,'XTick',1:A,     'XTickLabel', arrayfun(@(x)sprintf('%d',x), tauAxis,'uni',0));
set(ax,'YTick',1:B1,    'YTickLabel', yTickLbl);
xlabel('\tau'); ylabel('Config'); zlabel('Average rank');
title(sprintf('%s: Avg rank over %d functions', upper(string(algoMode)), cntF), 'FontWeight','normal');

% Annotate each bar with two-decimal value
hold on;
for i=1:B1
    for j=1:A
        zval = avgRank(i,j);
        text(j, i, zval+0.03, sprintf('%.2f', zval), ...
            'HorizontalAlignment','center', 'VerticalAlignment','bottom', ...
            'FontSize',9, 'Color',[0.1 0.1 0.1], 'FontWeight','bold', ...
            'Clipping','on');
    end
end
view([-32 28]); axis tight;

% ---- 3D surface ----
fig2 = figure('Color','w','Position',[220 80 980 640]);
[TAU, RHO] = meshgrid(tauAxis, rhoAxis);
surf(TAU, RHO, avgRank, 'EdgeAlpha',0.25, 'FaceAlpha',0.96, 'FaceColor','interp');
colormap(cm);
caxis([min(avgRank(:)) max(avgRank(:))]);
cb2 = colorbar; cb2.Label.String = 'Average rank (lower is better)';
ax2 = gca; grid on; box on; ax2.LineWidth=0.8; ax2.FontSize=11;
xlabel('\tau'); ylabel('Config'); zlabel('Average rank');
title(sprintf('%s: Avg-rank surface (best: \\tau=%d, \\rho=%.2f)', ...
      upper(string(algoMode)), best_tau, best_rho), 'FontWeight','normal');
set(ax2,'YTick',1:B1, 'YTickLabel', yTickLbl);
view([-38 30]); axis tight;
end

function save_png(fig, filename, dpi)
if ~ishandle(fig), return; end
try
    exportgraphics(fig, filename, 'Resolution', dpi);
catch
    try
        print(fig, filename, '-dpng', ['-r' num2str(dpi)]);
    catch
        saveas(fig, filename);
    end
end
end

function write_avg_rank_csv(csvFile, avgRank, tauAxis, rhoAxis)
T = array2table(avgRank, 'VariableNames', compose('tau_%g', tauAxis));
rho_labels = [{'Baseline'}, arrayfun(@(y)sprintf('%.2f',y), rhoAxis(2:end),'uni',0)];
T = addvars(T, rhoAxis(:), rho_labels(:), 'Before', 1, ...
    'NewVariableNames', {'rho_numeric','rho_label'});
writetable(T, csvFile);
end

function write_avg_rank_tex(texFile, avgRank, tauAxis, rhoAxis, best_tau, best_rho, cntF)
fid = fopen(texFile,'w');
fprintf(fid, '%% Auto-generated by make_cce_paper_tables\n');
fprintf(fid, '\\begin{table}[t]\\centering\\small\n');
fprintf(fid, '\\caption{Average rank across %d functions (rows: Baseline + $\\rho$, columns: $\\tau$). Lower is better.}\\label{tab:avg-rank}\n', cntF);
fprintf(fid, '\\begin{tabular}{l');
for j=1:numel(tauAxis), fprintf(fid,'c'); end
fprintf(fid, '}\\toprule\n');
fprintf(fid, 'Config ');
for j=1:numel(tauAxis), fprintf(fid, '& $\\tau=%d$ ', tauAxis(j)); end
fprintf(fid, '\\\\\\midrule\n');

for i=1:size(avgRank,1)
    if i==1
        fprintf(fid,'Baseline ');
    else
        fprintf(fid,'$\\rho=%.2f$ ', rhoAxis(i));
    end
    for j=1:size(avgRank,2)
        val = avgRank(i,j);
        fprintf(fid,'& %.2f ', val);
    end
    fprintf(fid,'\\\\\n');
end
fprintf(fid, '\\bottomrule\\end{tabular}\\end{table}\n');
fclose(fid);
end

function SumT = build_per_function_summary(All, FuncList, skipFuncs, tauAxis, rhoAxis)
A = numel(tauAxis); B = numel(rhoAxis)-1;
Func  = []; BestTau=[]; BestRho=[]; BaseMed=[]; CCEMed=[]; RelImp=[]; BaseTime=[]; CCETime=[];

for F = FuncList
    if ismember(F, skipFuncs), continue; end
    if F>numel(All) || isempty(All(F)) || ~isfield(All(F),'grid') || ~isfield(All(F).grid,'medGrid')
        continue;
    end
    medGrid = All(F).grid.medGrid;  % A×B expected
    timeGrid= All(F).grid.timeGrid; % A×B
    if isempty(medGrid) || isempty(timeGrid), continue; end
    if ~isequal(size(medGrid), [A, B])
        if isequal(size(medGrid), [B, A]), medGrid = medGrid.'; timeGrid = timeGrid.'; else, continue; end
    end
    baseMed   = All(F).baseline.med;
    baseTimes = All(F).baseline.time;   % vector of 20 runs
    baseMeanT = mean(baseTimes(:),'omitnan');

    % find best by median
    [bestMed, idx] = min(medGrid(:));
    [ia_best, ib_best] = ind2sub(size(medGrid), idx);
    btau = tauAxis(ia_best);
    brho = rhoAxis(ib_best+1);

    cceMeanT = timeGrid(ia_best, ib_best);

    relImp = 100*(baseMed - bestMed)/max(eps, baseMed);

    Func     = [Func ; F];
    BestTau  = [BestTau ; btau];
    BestRho  = [BestRho ; brho];
    BaseMed  = [BaseMed ; baseMed];
    CCEMed   = [CCEMed ; bestMed];
    RelImp   = [RelImp ; relImp];
    BaseTime = [BaseTime ; baseMeanT];
    CCETime  = [CCETime ; cceMeanT];
end

% Note: Removed Speedup from the table structure
SumT = table(Func, BestTau, BestRho, BaseMed, CCEMed, RelImp, BaseTime, CCETime);
end

function write_perfunc_longtable(texFile, T)
fid = fopen(texFile,'w');
fprintf(fid, '%% Auto-generated by make_cce_paper_tables\n');
fprintf(fid, '\\begin{center}\\small\n');
fprintf(fid, ['\\begin{longtable}{lcccccc}\n' ...  % Changed from 8 columns to 7
    '\\caption{Per-function grid best $(\\tau,\\rho)$ for CCEPSO and improvement over baseline ($D=100$, 20 runs). ' ...
    'Rel. improv. $=(\\mathrm{Base}-\\mathrm{CCE})/\\mathrm{Base}\\times 100\\%%$. ' ...
    'Full data in the repository.}\\label{tab:per-func-best}\\\\\n']);  % Removed speedup description
fprintf(fid, '\\toprule\n');
% Changed header to full names and removed Speedup column
fprintf(fid, 'Function & Best Configuration $(\\tau,\\rho)$ & Baseline Median $\\downarrow$ & CCE Median $\\downarrow$ & Relative Improvement (\\%%) $\\uparrow$ & Baseline Mean Time (s) & CCE Mean Time (s) \\\\\n');
fprintf(fid, '\\midrule\n');
fprintf(fid, '\\endfirsthead\n');
fprintf(fid, '\\toprule\n');
% Repeated header for continuation pages
fprintf(fid, 'Function & Best Configuration $(\\tau,\\rho)$ & Baseline Median $\\downarrow$ & CCE Median $\\downarrow$ & Relative Improvement (\\%%) $\\uparrow$ & Baseline Mean Time (s) & CCE Mean Time (s) \\\\\n');
fprintf(fid, '\\midrule\n');
fprintf(fid, '\\endhead\n');
fprintf(fid, '\\midrule\n');
fprintf(fid, '\\multicolumn{7}{r}{\\small Continued on next page}\\\\\n');  % Changed from 8 to 7
fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\endfoot\n');
fprintf(fid, '\\bottomrule\n');
fprintf(fid, '\\endlastfoot\n');

for r=1:height(T)
    % Removed the speedup column from the output
    fprintf(fid, 'F%02d & (%d, %.2f) & %.3e & \\textbf{%.3e} & \\textbf{%.1f} & %.2f & %.2f \\\\\n', ...
        T.Func(r), T.BestTau(r), T.BestRho(r), T.BaseMed(r), T.CCEMed(r), T.RelImp(r), T.BaseTime(r), T.CCETime(r));
end

fprintf(fid, '\\end{longtable}\n\\end{center}\n');
fclose(fid);
end

function export_excel_all(xlsxFile, All, FuncList, skipFuncs, tauAxis, rhoAxis, avgRank, SumT)
% Remove existing file to avoid sheet append collisions
if exist(xlsxFile,'file'), delete(xlsxFile); end

% Sheet: AvgRank
writematrix(avgRank, xlsxFile, 'Sheet','AvgRank','Range','B2');
writecell([{' '}, compose('tau_%g', tauAxis)], xlsxFile, 'Sheet','AvgRank','Range','B1');
rho_labels = [{'Baseline'}, arrayfun(@(y)sprintf('%.2f',y), rhoAxis(2:end),'uni',0)];
writecell([{'Config'}; rho_labels(:)], xlsxFile, 'Sheet','AvgRank','Range','A1');

% Sheet: Axes (both numeric & labels)
writecell({'tauAxis'}, xlsxFile,'Sheet','Axes','Range','A1');
writematrix(tauAxis(:), xlsxFile,'Sheet','Axes','Range','A2');
writecell({'rhoAxis (numeric)'}, xlsxFile,'Sheet','Axes','Range','C1');
writematrix(rhoAxis(:), xlsxFile,'Sheet','Axes','Range','C2');
writecell({'rhoAxis (labels)'}, xlsxFile,'Sheet','Axes','Range','E1');
writecell(rho_labels(:), xlsxFile,'Sheet','Axes','Range','E2');

% Sheet: PerFuncBest
writetable(SumT, xlsxFile, 'Sheet','PerFuncBest');

% Per-function sheets: medGrid/timeGrid and baseline basic stats
A = numel(tauAxis); B = numel(rhoAxis)-1;
for F = FuncList
    if ismember(F, skipFuncs), continue; end
    if F>numel(All) || isempty(All(F)) || ~isfield(All(F),'grid') || ~isfield(All(F).grid,'medGrid')
        continue;
    end
    medGrid = All(F).grid.medGrid;
    timeGrid= All(F).grid.timeGrid;
    if isempty(medGrid) || isempty(timeGrid), continue; end
    if ~isequal(size(medGrid), [A, B])
        if isequal(size(medGrid), [B, A]), medGrid = medGrid.'; timeGrid = timeGrid.'; else, continue; end
    end
    sh = sprintf('F%02d', F);

    % --- medGrid sheet ---
    writecell({'rho'}, xlsxFile, 'Sheet',[sh '_medGrid'],'Range','A1');
    rho_row_labels = arrayfun(@(y)sprintf('%.2f',y), rhoAxis(2:end),'uni',0)';  % B×1
    writecell(rho_row_labels, xlsxFile, 'Sheet',[sh '_medGrid'],'Range','A2');
    writecell([{' '}, compose('tau_%g', tauAxis)], xlsxFile, 'Sheet',[sh '_medGrid'],'Range','B1');
    writematrix(medGrid, xlsxFile, 'Sheet', [sh '_medGrid'], 'Range','B2');

    % --- timeGrid sheet ---
    writecell({'rho'}, xlsxFile, 'Sheet',[sh '_timeGrid'],'Range','A1');
    writecell(rho_row_labels, xlsxFile, 'Sheet',[sh '_timeGrid'],'Range','A2');
    writecell([{' '}, compose('tau_%g', tauAxis)], xlsxFile, 'Sheet',[sh '_timeGrid'],'Range','B1');
    writematrix(timeGrid, xlsxFile, 'Sheet', [sh '_timeGrid'], 'Range','B2');

    % --- Baseline stats ---
    baseMed   = All(F).baseline.med;
    baseTimes = All(F).baseline.time(:);
    Tbas = table(baseMed, mean(baseTimes,'omitnan'),'VariableNames',{'BaseMedian','BaseMeanTime'});
    writetable(Tbas, xlsxFile, 'Sheet', [sh '_baseline']);
end
end