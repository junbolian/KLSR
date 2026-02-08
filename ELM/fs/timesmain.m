%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%% 添加路径文件 
addpath(genpath(pwd));

%% 导入数据
res = xlsread('lung-cancer.xlsx');
rng(42, 'twister');                     % 随机种子 确保实验可重复

% 设置运行次数
num_runs = 10;
train_accuracies = zeros(1, num_runs);
test_accuracies = zeros(1, num_runs);

for run = 1:num_runs
    %% 分析数据
    num_class = length(unique(res(:, end)));  % 类别数（Excel最后一列放类别）
    num_res = size(res, 1);                   % 样本数（每一行，是一个样本）
    num_size = 0.7;                           % 训练集占数据集的比例
    res = res(randperm(num_res), :);          % 打乱数据集

    %% 设置变量存储数据
    P_train = []; P_test = [];
    T_train = []; T_test = [];

    %% 划分数据集
    for i = 1:num_class
        mid_res = res((res(:, end) == i), :);           % 循环取出不同类别的样本
        mid_size = size(mid_res, 1);                    % 得到不同类别样本个数
        mid_tiran = round(num_size * mid_size);         % 得到该类别的训练样本个数

        P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % 训练集输入
        T_train = [T_train; mid_res(1: mid_tiran, end)];              % 训练集输出

        P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % 测试集输入
        T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % 测试集输出
    end

    %% 数据转置
    P_train = P_train'; P_test = P_test';

    %% 数据归一化
    [p_train, ps_input] = mapminmax(P_train, 0, 1);
    p_test  = mapminmax('apply', P_test, ps_input);
    t_train = T_train';
    t_test  = T_test';

    %% 设置适应度函数
    fobj = @(x)getObjValue(x, p_train, t_train, p_test, t_test);

    %% 设置优化算法参数
    SearchAgents_no = 50;                    % 种群个数
    Max_iter = 50;                           % 迭代次数
    dim = size(res, 2) - 1;                  % 维度
    lb = 0 .* ones(1, dim);                  % 下限值
    ub = 1 .* ones(1, dim);                  % 上限值

    %% 调用算法
    [Best_score, Best_pos, Curve] = CMAES(SearchAgents_no, Max_iter, lb, ub, dim, fobj);

    %% 得到最佳特征索引
    id = round(Best_pos);
    idx = find(id == 1);

    %% 数据的参数
    p_train = p_train(idx, :);
    p_test = p_test(idx, :);

    %% 创建模型
    num_hiddens = 50;        % 隐藏层节点个数
    activate_model = 'sig';  % 激活函数
    [IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, num_hiddens, activate_model, 1);

    %% 仿真测试
    T_sim1 = elmpredict(p_train, IW, B, LW, TF, TYPE);
    T_sim2 = elmpredict(p_test, IW, B, LW, TF, TYPE);

    %% 数据排序
    [T_train, index_1] = sort(T_train);
    [T_test, index_2] = sort(T_test);
    T_sim1 = T_sim1(index_1);
    T_sim2 = T_sim2(index_2);

    %% 数据转置
    T_sim1 = T_sim1';
    T_sim2 = T_sim2';

    %% 适应度值
    error1 = sum((T_sim1 == T_train)) / length(T_train);
    error2 = sum((T_sim2 == T_test)) / length(T_test);

    % 存储结果
    train_accuracies(run) = error1 * 100;
    test_accuracies(run) = error2 * 100;
end

%% 计算并显示平均值
mean_train_accuracy = mean(train_accuracies);
mean_test_accuracy = mean(test_accuracies);
disp(['ELM分类器训练集平均准确率为：', num2str(mean_train_accuracy), '%']);
disp(['ELM分类器测试集平均准确率为：', num2str(mean_test_accuracy), '%']);
