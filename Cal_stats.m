% Author: Junbo Jacob Lian

function [Results, wilcoxon_test, friedman_p_value] = Cal_stats(Optimal_results, time_record)
% ------------------------------------------------------------
% Results 索引:
%   1  Algorithm name
%   2  Mean convergence curve
%   3  worst  (max over runs)
%   4  best   (min over runs)
%   5  std    (std over runs)
%   6  mean   (mean over runs)
%   7  time   (mean seconds per run)
% ------------------------------------------------------------
num_alg = size(Optimal_results,2);
num_run = size(Optimal_results{3,1},1);

Results = cell(7,num_alg);

for k = 1:num_alg
    Results{1,k} = Optimal_results{1,k};
    Results{2,k} = mean(Optimal_results{2,k},1);
    Results{3,k} = max (Optimal_results{3,k});  % worst
    Results{4,k} = min (Optimal_results{3,k});  % best
    Results{5,k} = std (Optimal_results{3,k});  % std
    Results{6,k} = mean(Optimal_results{3,k});  % mean
    Results{7,k} = mean(time_record(:,k));      % average time
end

% ---------- Wilcoxon (baseline = 第一个算法) ----------
baseline = Optimal_results{3,1};
for k = 2:num_alg
    target = Optimal_results{3,k};
    wilcoxon_test.signed_p_value(k-1)  = signrank(baseline,target);
    wilcoxon_test.ranksum_p_value(k-1) = ranksum (baseline,target);
end

% ---------- Friedman ----------
test_mat = zeros(num_run,num_alg);
for k = 1:num_alg
    test_mat(:,k) = Optimal_results{3,k};
end
friedman_p_value = friedman(test_mat,1,'off');
end
