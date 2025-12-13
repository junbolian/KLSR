% ============================================================
%  Author: Junbo Jacob Lian
%  Average ranks over functions for Baseline + CCE grid
%  -> Exactly 13 bars when A=numel(tau), B=numel(rho): 1 + A*B
%  Figures:
%    - 3D bar (13 bars; bar-top labels with 2 decimals; Baseline shown once)
%    - 3D surface over tau×rho grid (no baseline)
%  Data export:
%    - <mat>-AvgRankFlat.csv/.tex (13 configs)
%    - <mat>-AvgRank3D.xlsx (AvgRank/Axes/Summary/PerFuncRank/PerFuncMedian/PerFuncBest)
%    - <mat>-PerFuncBest.csv/.tex
%  All writers have permission-safe fallback to tempdir.
%  Usage:
%    plot_avg_rank3D('results_CCE_PSO_CEC2017_D100_compare.mat')
% ============================================================

function plot_avg_rank3D(matfile)
% Average-rank over functions with PSO baseline INCLUDED in the same ranking set.
% Left: single baseline bar. Right: 3D bar grid for 12 (tau,rho) configs.
% Exports: Rank.png + PDF + CSV + LaTeX + Excel (all include baseline).

clc; close all;
S = load(matfile);

% ---- Required fields ----
All       = S.All;
tau_grid  = S.tau_grid(:)';          % 1×A
rho_grid  = S.rho_grid(:)';          % 1×B

% Optional
if isfield(S,'FuncList'),   FuncList = S.FuncList;   else, FuncList = 1:numel(All); end
if isfield(S,'skipFuncs'),  skipFuncs = S.skipFuncs; else, skipFuncs = [];          end
if isfield(S,'algoMode'),   algoMode = S.algoMode;   else, algoMode = 'ALGO';       end

tauAxis = tau_grid;                  % 1×A
rhoAxis = [0, rho_grid];             % 1×(B+1), baseline=0 at row 1
A = numel(tauAxis);
B = numel(rhoAxis)-1;

% ---- Accumulators (global ranking of 13 configs) ----
sumBaseRank = 0;                     % scalar
sumGridRank = zeros(B, A);           % B×A (rows=rho>0, cols=tau)
cntF        = 0;                     % number of functions used
usedFuncs   = [];

% Optional cubes (per-function rank/value for export)
rankGridCube = []; baseRankVec = [];
valGridCube  = []; baseValVec  = [];
try
  rankGridCube = nan(B, A, 512); baseRankVec = nan(512,1);
  valGridCube  = nan(B, A, 512); baseValVec  = nan(512,1);
catch, end

for F = FuncList
    if ismember(F, skipFuncs), continue; end
    if F>numel(All) || isempty(All(F)) || ~isfield(All(F),'grid') || ~isfield(All(F).grid,'medGrid'), continue; end

    medGrid = All(F).grid.medGrid;       % A×B (rows=tau, cols=rho)
    if isempty(medGrid), continue; end
    if ~isequal(size(medGrid), [A,B])
        if isequal(size(medGrid), [B,A]), medGrid = medGrid.'; else, continue; end
    end
    baseMed = All(F).baseline.med;
    if isempty(baseMed) || ~isscalar(baseMed) || ~isfinite(baseMed), continue; end

    % ---- Rank across ALL 13 configs together (baseline + 12 grid) ----
    vals13   = [baseMed ; medGrid(:)];                  % (1+AB)×1 (column-major)
    if exist('tiedrank','file')
        ranks13 = tiedrank(vals13);                     % lower is better -> rank 1
    else
        [~,ord] = sort(vals13,'ascend'); ranks13 = zeros(size(vals13)); ranks13(ord) = 1:numel(vals13);
    end
    baseRank  = ranks13(1);
    gridRanks = reshape(ranks13(2:end), [A B]).';       % -> B×A (rows=rho>0, cols=tau)
    gridVals  = reshape(vals13(2:end),  [A B]).';

    % ---- accumulate ----
    sumBaseRank = sumBaseRank + baseRank;
    sumGridRank = sumGridRank + gridRanks;
    cntF        = cntF + 1;
    usedFuncs(end+1) = F; %#ok<AGROW>
    if ~isempty(rankGridCube)
        rankGridCube(:,:,cntF) = gridRanks;  baseRankVec(cntF) = baseRank;
        valGridCube(:,:,cntF)  = gridVals;   baseValVec(cntF)  = baseMed;
    end
end

if cntF==0, error('No valid functions to aggregate.'); end
avgBaseRank = sumBaseRank / cntF;     % scalar
avgRankGrid = sumGridRank / cntF;     % B×A

% ---- Best (tau,rho) among grid (exclude baseline) ----
[bestVal, bestIdx] = min(avgRankGrid(:));
[iRho, jTau] = ind2sub(size(avgRankGrid), bestIdx);
best_rho = rhoAxis(iRho+1); best_tau = tauAxis(jTau);

% ---- Labels for flat export ----
cfgLabels = cell(1 + A*B, 1);
cfgLabels{1} = 'Baseline';
k = 2;
for j = 1:A
    for i = 2:(B+1)
        cfgLabels{k} = sprintf('\\tau=%d, \\rho=%.2f', tauAxis(j), rhoAxis(i));
        k = k + 1;
    end
end

% ---- Aesthetics ----
set(0,'DefaultAxesFontName','Helvetica','DefaultTextFontName','Helvetica');
cmap = natureBlue(256);
zmin = min([avgBaseRank; avgRankGrid(:)]);
zmax = max([avgBaseRank; avgRankGrid(:)]);

% ===================== Combined Figure =====================
tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact','TileIndexing','rowmajor');
fig = gcf; set(fig,'Color','w','Position',[150 80 1240 720],'Renderer','painters');

% ------ Left: single baseline bar ------
nexttile(1);
tcol = max(0,min(1,(avgBaseRank - zmin) / max(eps, zmax - zmin)));
bColor = cmap( max(1, round(tcol*(size(cmap,1)-1))+1 ), :);
bar(1, avgBaseRank, 0.55, 'FaceColor', bColor, 'EdgeColor',[0.35 0.35 0.35], 'LineWidth',0.8);
ylim([max(0,zmin-0.15*(zmax-zmin))  zmax+0.15*(zmax-zmin)]);
xlim([0 2]); grid on; box on;
axL = gca; axL.LineWidth=0.9; axL.FontSize=12;
set(axL,'XTick',1,'XTickLabel',{'Baseline'});
ylabel('Average rank');
title(sprintf('%s: Baseline (avg over %d functions)', upper(string(algoMode)), cntF), 'FontWeight','normal');
text(1, avgBaseRank + 0.03*(zmax - zmin + eps), sprintf('%.2f', avgBaseRank), ...
    'HorizontalAlignment','center','VerticalAlignment','bottom', ...
    'FontSize',11,'FontWeight','bold','BackgroundColor','w','Margin',1,'EdgeColor','none');

% ------ Right: 3D bar for tau×rho grid ------
nexttile(2);
bh = bar3(avgRankGrid, 'detached');    % rows=rho>0, cols=tau
for kk = 1:numel(bh), set(bh(kk),'EdgeColor',[0.35 0.35 0.35],'LineWidth',0.6); end
colormap(cmap); caxis([zmin zmax]);
cb = colorbar; cb.Label.String = 'Average rank (lower is better)';
grid on; box on; axR = gca; axR.LineWidth=0.9; axR.FontSize=12;
set(axR,'XTick',1:A, 'XTickLabel', arrayfun(@(x)sprintf('%d',x), tauAxis,'uni',0));
set(axR,'YTick',1:B, 'YTickLabel', arrayfun(@(y)sprintf('%.2f',y), rhoAxis(2:end),'uni',0));
xlabel('\tau'); ylabel('\rho'); zlabel('Average rank');
title(sprintf('CCEPSO configs (best: \\tau=%d, \\rho=%.2f)', best_tau, best_rho), 'FontWeight','normal');
view([-35 26]); axis tight; drawnow;
hold on;
for ii = 1:B
    for jj = 1:A
        z = avgRankGrid(ii,jj);
        text(jj, ii, z + 0.03*(zmax - zmin + eps), sprintf('%.2f', z), ...
            'HorizontalAlignment','center','VerticalAlignment','bottom', ...
            'FontSize',10,'FontWeight','bold','BackgroundColor','w', ...
            'Margin',1,'EdgeColor','none','Clipping','off');
    end
end
sgtitle(sprintf('%s: Average rank over %d functions (baseline left + 12 configs right)', ...
        upper(string(algoMode)), cntF), 'FontWeight','normal','FontSize',12);

% ===================== Save main fig =====================
[pth, nam] = fileparts(matfile); if isempty(pth), pth = pwd; end
pngMain = fullfile(pth, 'Rank.png');   % manuscript alias
pdfMain = fullfile(pth, [nam '-Rank.pdf']);
save_hq(fig, pngMain, pdfMain);

% ===================== Save CSV & LaTeX =====================
csvFile  = fullfile(pth, [nam '-AvgRank13.csv']);  % 13 configs (incl. baseline)
texFile  = fullfile(pth, [nam '-AvgRank13.tex']);

% CSV flat (13 rows)
flatVals = [avgBaseRank ; avgRankGrid(:)];
Tflat = table((1:numel(flatVals))', cfgLabels, flatVals, 'VariableNames', {'idx','config','avg_rank'});
safe_write_table(Tflat, csvFile);

% LaTeX: baseline + grid matrix
fid = fopen(texFile, 'w');
fprintf(fid, '%% Auto-generated by plot_avg_rank3D (13-config ranking incl. baseline)\n');
fprintf(fid, '\\begin{table}[t]\\centering\\small\\caption{Average rank across %d functions. Left: baseline. Right: grid rows $\\rho\\in\\{%.2f,\\dots,%.2f\\}$, columns $\\tau\\in\\{%s\\}$. Lower is better. Best at $\\tau=%d,\\,\\rho=%.2f$.}\\label{tab:avg-rank-13}\n', ...
        cntF, rhoAxis(2), rhoAxis(end), strjoin(arrayfun(@num2str,tauAxis,'uni',0), ','), best_tau, best_rho);
fprintf(fid, '\\begin{tabular}{l c}\\toprule Baseline & %.2f \\\\ \\bottomrule\\end{tabular}\\\\[2pt]\n', avgBaseRank);
fprintf(fid, '\\begin{tabular}{l'); for j=1:A, fprintf(fid,'c'); end
fprintf(fid, '}\\toprule\n$\\rho$ ');
for j=1:A, fprintf(fid, '& $\\tau=%d$ ', tauAxis(j)); end
fprintf(fid, '\\\\\\midrule\n');
for i=1:B
    fprintf(fid, '%.2f ', rhoAxis(i+1));
    for j=1:A
        val = avgRankGrid(i,j);
        if i==iRho && j==jTau, fprintf(fid, '& \\textbf{%.2f} ', val);
        else,                  fprintf(fid, '& %.2f ', val);
        end
    end
    fprintf(fid, '\\\\\n');
end
fprintf(fid,'\\bottomrule\\end{tabular}\\end{table}\n');
fclose(fid);

% ===================== Excel (baseline + grid) =====================
xlsxFile = fullfile(pth, [nam '-AvgRank13.xlsx']);
% Sheet: AvgRank (matrix with baseline as separate single cell)
try
    writematrix(avgBaseRank, xlsxFile, 'Sheet','AvgRank','Range','B2');   % baseline
    writematrix(tauAxis(:)', xlsxFile, 'Sheet','AvgRank','Range','C1');   % header tau
    writematrix(rhoAxis(2:end).', xlsxFile, 'Sheet','AvgRank','Range','B3'); % header rho
    writematrix(avgRankGrid,  xlsxFile, 'Sheet','AvgRank','Range','C3');  % grid
catch
    warning('Excel write failed (permission?). Writing to tempdir instead.');
    xlsxFile = fullfile(tempdir, [nam '-AvgRank13.xlsx']);
    writematrix(avgBaseRank, xlsxFile, 'Sheet','AvgRank','Range','B2');
    writematrix(tauAxis(:)', xlsxFile, 'Sheet','AvgRank','Range','C1');
    writematrix(rhoAxis(2:end).', xlsxFile, 'Sheet','AvgRank','Range','B3');
    writematrix(avgRankGrid,  xlsxFile, 'Sheet','AvgRank','Range','C3');
end

% Sheet: Summary
writecell({'algoMode','num_functions','best_tau','best_rho','best_avg_rank','baseline_avg_rank'}, xlsxFile,'Sheet','Summary','Range','A1');
writecell({upper(string(algoMode)), cntF, best_tau, best_rho, bestVal, avgBaseRank},      xlsxFile,'Sheet','Summary','Range','A2');

% Sheet: Axes
writecell({'tauAxis'}, xlsxFile,'Sheet','Axes','Range','A1'); writematrix(tauAxis(:), xlsxFile,'Sheet','Axes','Range','A2');
writecell({'rhoAxis (incl. 0=baseline)'}, xlsxFile,'Sheet','Axes','Range','C1'); writematrix(rhoAxis(:), xlsxFile,'Sheet','Axes','Range','C2');

% Sheet: PerFuncRank (baseline + grid)
LONG = {'Func','tau','rho','rank','value','isBaseline'}; row = 2;
for idxF = 1:cntF
    F = usedFuncs(idxF);
    if ~isempty(rankGridCube)
        Rg = rankGridCube(:,:,idxF); Vg = valGridCube(:,:,idxF);
        Rb = baseRankVec(idxF);      Vb = baseValVec(idxF);
    else
        % recompute for this F
        medGrid = All(F).grid.medGrid; baseMed = All(F).baseline.med;
        if ~isequal(size(medGrid), [A,B]), medGrid = medGrid.'; end
        vals13   = [baseMed ; medGrid(:)];
        if exist('tiedrank','file'), ranks13 = tiedrank(vals13);
        else, [~,ord]=sort(vals13,'ascend'); ranks13=zeros(size(vals13)); ranks13(ord)=1:numel(vals13);
        end
        Rb = ranks13(1); Vb = baseMed;
        Rg = reshape(ranks13(2:end), [A B]).';  Vg = reshape(vals13(2:end), [A B]).';
    end
    % baseline row (tau=0, rho=0)
    LONG(row,:) = {F, 0, 0, Rb, Vb, 1}; row = row + 1;
    % grid rows
    for i=1:B
        for j=1:A
            LONG(row,:) = {F, tauAxis(j), rhoAxis(i+1), Rg(i,j), Vg(i,j), 0}; %#ok<AGROW>
            row = row + 1;
        end
    end
end
writecell(LONG, xlsxFile, 'Sheet','PerFuncRank','Range','A1');

% Sheet: PerFuncMedian (wide; first column baseline, then tau×rho)
Hdr = [{'Func','Baseline'}, arrayfun(@(j)arrayfun(@(i)sprintf('tau=%d|rho=%.2f',tauAxis(j),rhoAxis(i)), 2:(B+1), 'uni',0), 'uni',0)];
Hdr = [Hdr{:}];
writecell(Hdr, xlsxFile, 'Sheet','PerFuncMedian','Range','A1');
r0 = 2;
for idxF=1:cntF
    F = usedFuncs(idxF);
    medGrid = All(F).grid.medGrid; baseMed = All(F).baseline.med;
    if ~isequal(size(medGrid), [A,B]), medGrid = medGrid.'; end
    rowVals = [baseMed, reshape(medGrid.', 1, [])];   % 1 + (B*A)
    writecell({F}, xlsxFile, 'Sheet','PerFuncMedian','Range',sprintf('A%d', r0));
    writematrix(rowVals, xlsxFile, 'Sheet','PerFuncMedian','Range',sprintf('B%d', r0));
    r0 = r0 + 1;
end

fprintf('Saved:\n  Rank fig : %s\n  Rank PDF : %s\n  CSV      : %s\n  LaTeX    : %s\n  Excel    : %s\n', ...
        pngMain, pdfMain, csvFile, texFile, xlsxFile);
fprintf('Best grid config at tau=%d, rho=%.2f (avg rank=%.2f). Baseline avg rank=%.2f\n', ...
        best_tau, best_rho, bestVal, avgBaseRank);
end

% ---------- helpers ----------
function safe_write_table(T, file)
try
    writetable(T, file);
catch
    warning('Could not write %s (permission?). Writing to tempdir instead.', file);
    [p, n, ~] = fileparts(file);
    fallback = fullfile(tempdir, [n '_fallback.csv']);
    writetable(T, fallback);
end
end

function save_hq(fig, png, pdf)
if ~ishghandle(fig), warning('Figure invalid for %s', png); return; end
drawnow;
try
    if exist('exportgraphics','file')==2
        exportgraphics(fig, png, 'Resolution',700,'BackgroundColor','white');
        if nargin>2 && ~isempty(pdf)
            exportgraphics(fig, pdf, 'BackgroundColor','white');
        end
    else
        set(fig,'InvertHardcopy','off'); set(fig,'PaperPositionMode','auto');
        print(fig, png, '-dpng','-r600');
        if nargin>2 && ~isempty(pdf)
            print(fig, pdf, '-dpdf','-painters');
        end
    end
catch ME
    warning('Save failed (%s). Fallback to getframe.', ME.message);
    fr = getframe(fig);
    imwrite(fr.cdata, png);
end
end

function C = natureBlue(n)
lo = [0.96 0.98 1.00];   % light bluish white
hi = [0.16 0.33 0.58];   % deep desaturated blue
t  = linspace(0,1,n)'; C = lo.*(1-t) + hi.*t;
end
