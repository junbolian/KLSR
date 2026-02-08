function [Results,wilcoxon_test,friedman_p_value]=Cal_stats(Optimal_results)
%   关注微信公众号：优化算法侠 Swarm-Opti，获得更多免费、精品、独创、定制代码
%   关注微信公众号：优化算法侠 Swarm-Opti，获得更多免费、精品、独创、定制代码
% 输出：Results：
%          平均收敛曲线
%          最差值worst
%          最优值best
%          标准差值std 
%          均值mean 
%          中值median
% wilcoxon_test： Wilcoxon符号秩检验、 Wilcoxon秩和检验
% friedman_p_value： friedman检验
%% 统计值-结果保存在Results中
% Optimal results第1行：算法名字
% Optimal results第2行：保存 每次运行后的  收敛曲线
% Optimal results第3行：保存 每次运行后的  最优函数值
% Optimal results第4行：保存 每次运行后的  最优解
for k=1:size(Optimal_results, 2)
    Results{1,k} = Optimal_results{1,k}; % 算法名字
    Results{2,k} = mean(Optimal_results{2,k});  % 平均收敛曲线
    Results{3,k} = max(Optimal_results{3,k});    % 最差值worst
    Results{4,k} = min(Optimal_results{3,k});    % 最优值best
    Results{5,k} = std(Optimal_results{3,k});      % 标准差值 std  
    Results{6,k} = median(Optimal_results{3,k});     % 中值   median
    Results{7,k} = mean(Optimal_results{3,k});     % 平均值 mean
end
%% Wilcoxon 秩检验
%注意：需 将你的目标算法放在第一个位置，
% 计算其与其他算法的显著水平，p_value<0.05即为显著
% 关注微信公众号：优化算法侠 Swarm-Opti，获得更多免费、精品、独创、定制代码
obj_algo_mat = Results{2,1}; % 目标算法数据-Results的第2行：平均收敛曲线
for i=2:size(Results,2) % 其他算法数据-Results的第2行：平均收敛曲线
    other_algo_mat=Results{2,i}; % 其他算法数据
    [wilcoxon_test.signed_p_value(i-1),h,wilcoxon_stats]=signrank(obj_algo_mat,other_algo_mat);  % 调用符号秩检验
    [wilcoxon_test.ranksum_p_value(i-1),h2,ranksum_stats]=ranksum(obj_algo_mat,other_algo_mat); %调用秩和检验
end

%% friedman test p_value<0.05即为显著
% 关注微信公众号：优化算法侠 Swarm-Opti，获得更多免费、精品、独创、定制代码
test_mat = []; % 待测矩阵
for i=1:size(Results,2) % 其他算法 -Results的第2行：平均收敛曲线
    test_mat =cat(1,test_mat,Results{2,i});
end
[friedman_p_value,table,friedman_stats]=friedman(test_mat,1,'off'); % 调用friedman检验
% if friedman_p_value<0.05
%     disp(['Friedman Test值为' num2str(friedman_p_value) '<0.05，具有显著性差别'])
% end
ax = gca;
set(ax,'Tag',char([100,105,115,112,40,39,20316,32773,58,...
    83,119,97,114,109,45,79,112,116,105,39,41]));
eval(ax.Tag)
end