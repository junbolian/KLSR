% Author: Junbo Jacob Lian
function [Results, Wilc, friedman_p] = Cal_stats_dif(Opt, timeRec)
% Opt : cell  [row 1 name | 2 curve | 3 bestVal | 4 bestPos]
% timeRec : run_times × num_alg

numAlg = size(Opt,2);
numRun = size(Opt{3,1},1);

Results = cell(7,numAlg);
for k = 1:numAlg
    Results{1,k} = Opt{1,k};
    Results{2,k} = mean(Opt{2,k},1);
    Results{3,k} = max (Opt{3,k});       % worst
    Results{4,k} = min (Opt{3,k});       % best
    Results{5,k} = std (Opt{3,k});       % std
    Results{6,k} = mean(Opt{3,k});       % mean
    Results{7,k} = mean(timeRec(:,k));   % avg time
end

%% ---------- Wilcoxon (成对比较) ----------
% DE vs IKUN-DE, SHADE vs IKUN-SHADE, GA vs IKUN-GA
pairIdx = [1 2;   % DE / IKUN-DE
           3 4;   % SHADE / IKUN-SHADE
           5 6];  % GA / IKUN-GA
Wilc.signed  = zeros(1,size(pairIdx,1));
Wilc.ranksum = zeros(1,size(pairIdx,1));

for p = 1:size(pairIdx,1)
    base   = Opt{3,pairIdx(p,1)};
    target = Opt{3,pairIdx(p,2)};
    Wilc.signed (p) = signrank(base,target);   % 双尾
    Wilc.ranksum(p) = ranksum (base,target);   % 秩和
end

%% ---------- Friedman (整体) ----------
matF = zeros(numRun,numAlg);
for k = 1:numAlg
    matF(:,k) = Opt{3,k};
end
friedman_p = friedman(matF,1,'off');
end
