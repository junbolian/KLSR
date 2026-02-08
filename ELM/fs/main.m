%%
% 机器不会学习 
% 面包多主页：https://mbd.pub/o/curry/work
%%
%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行

%% 添加路径文件 
addpath(genpath(pwd));

%%  导入数据
res = xlsread('数据集.xlsx');
rng(42,'twister');                        % 随机种子 确保实验可重复

%%  分析数据
num_class = length(unique(res(:, end)));  % 类别数（Excel最后一列放类别）
num_res = size(res, 1);                   % 样本数（每一行，是一个样本）
num_size = 0.7;                           % 训练集占数据集的比例
res = res(randperm(num_res), :);          % 打乱数据集（不打乱数据时，注释该行）

%%  设置变量存储数据
P_train = []; P_test = [];
T_train = []; T_test = [];

%%  划分数据集
for i = 1 : num_class
    mid_res = res((res(:, end) == i), :);           % 循环取出不同类别的样本
    mid_size = size(mid_res, 1);                    % 得到不同类别样本个数
    mid_tiran = round(num_size * mid_size);         % 得到该类别的训练样本个数

    P_train = [P_train; mid_res(1: mid_tiran, 1: end - 1)];       % 训练集输入
    T_train = [T_train; mid_res(1: mid_tiran, end)];              % 训练集输出

    P_test  = [P_test; mid_res(mid_tiran + 1: end, 1: end - 1)];  % 测试集输入
    T_test  = [T_test; mid_res(mid_tiran + 1: end, end)];         % 测试集输出
end

%%  数据转置
P_train = P_train'; P_test = P_test';

%%  得到训练集和测试样本个数  
M = size(P_train, 2);
N = size(P_test , 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train, 0, 1);
p_test  = mapminmax('apply', P_test, ps_input);
t_train = T_train';
t_test  = T_test';

%%  设置适应度函数  
fobj = @(x)getObjValue(x, p_train, t_train, p_test, t_test);

%%  设置优化算法参数
SearchAgents_no = 30;                    %  种群个数
Max_iter = 10;                           %  迭代次数
dim = size(res,2) - 1;                   %  维度
lb = 0.*ones(1,dim);                     %  下限值，特征选择的参数
ub = 1.*ones(1,dim);                     %  上限值，特征选择的参数

%%  调用算法 
[Best_score, Best_pos, Curve] = PO(SearchAgents_no,Max_iter,lb,ub,dim,fobj);
% 
% %%  绘制进化曲线 
% figure
% plot(Curve, 'linewidth',1.5)
% xlabel('迭代次数')
% ylabel('适应度值')
% title('收敛曲线图')
% set(gcf,'color','w')

%%  得到最佳特征索引  
id =  round(Best_pos);
idx = find(id==1);

%%  数据的参数
p_train = p_train(idx,:);
p_test = p_test(idx,:);

%%  创建模型
num_hiddens = 50;        % 隐藏层节点个数
activate_model = 'sig';  % 激活函数
[IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, num_hiddens, activate_model, 1);

%%  仿真测试
T_sim1 = elmpredict(p_train, IW, B, LW, TF, TYPE);
T_sim2 = elmpredict(p_test , IW, B, LW, TF, TYPE);

%%  数据排序
[T_train , index_1] = sort(T_train);
[T_test , index_2] = sort(T_test);
T_sim1 = T_sim1(index_1);
T_sim2 = T_sim2(index_2);

%%  数据转置
T_sim1 = T_sim1';
T_sim2 = T_sim2';

%%  适应度值 
error1 = sum((T_sim1 == T_train)) / length(T_train);
error2 = sum((T_sim2 == T_test)) / length(T_test);

% %%  绘图
% figure
% plot(1: M, T_train, 'r-*', 1: M, T_sim1, 'b-o', 'LineWidth', 1)
% legend('真实值', '预测值')
% xlabel('预测样本')
% ylabel('预测结果')
% string = {'训练集预测结果对比'; ['准确率=' num2str(error1) '%']};
% title(string)
% grid
% 
% figure
% plot(1: N, T_test, 'r-*', 1: N, T_sim2, 'b-o', 'LineWidth', 1)
% legend('真实值', '预测值')
% xlabel('预测样本')
% ylabel('预测结果')
% string = {'测试集预测结果对比'; ['准确率=' num2str(error2) '%']};
% title(string)
% grid
% 
% 
% %%  混淆矩阵
% figure
% cm = confusionchart(T_train, T_sim1);
% cm.Title = 'Confusion Matrix for Train Data';
% cm.ColumnSummary = 'column-normalized';
% cm.RowSummary = 'row-normalized';
% 
% figure
% cm = confusionchart(T_test, T_sim2);
% cm.Title = 'Confusion Matrix for Test Data';
% cm.ColumnSummary = 'column-normalized';
% cm.RowSummary = 'row-normalized';


%%  输出准确率 
disp('特征选择结果（0表示该特征被抛弃, 1表示该特征被选择）：')
A_A = round(Best_pos(1:end));
disp(num2str(A_A))
total_sum = sum(A_A(:));
disp('特征选择总数：')
disp(num2str(total_sum))
disp('特征选择的索引：')
disp(num2str(idx))
disp(['ELM分类器训练集准确率为：',num2str(error1*100),'%']);
disp(['ELM分类器测试集准确率为：',num2str(error2*100),'%']);