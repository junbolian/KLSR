
function [error,T_sim] = getObjValue(x,p_train,t_train,p_test,t_test)

    temp=clock;
    temp=sum(temp(4:6))*sum(temp(2:3));
    temp=round(temp/10);
    setdemorandstream(temp);  % 此行代码用于生成随机数种子，确保结果可以复现
    
    id =  round(x);
    idx = find(id==1);

    if isempty(idx) == 1      % 表述没有任何特征被选择
        error = 10000;        % 设置一个比较大的数，表示效果很差！
    else
        %%  数据的参数
        p_train = p_train(idx,:);
        p_test = p_test(idx,:);
 
        %%  创建模型
        num_hiddens = 50;        % 隐藏层节点个数
        activate_model = 'sig';  % 激活函数
        [IW, B, LW, TF, TYPE] = elmtrain(p_train, t_train, num_hiddens, activate_model, 1);
        
        %%  仿真测试
        T_sim = elmpredict(p_test , IW, B, LW, TF, TYPE);

        %%  获取适应度
        accuracy = sum(T_sim == t_test) / length(t_test);   % 计算预测准确率
        
        %% 分类错误率
        error = 1 - accuracy;
        
        % %% 特征选择的目标函数
        % p = 0.7;  % 是权重系数 可以调整
        % 
        % error = (1-p) * E + p * (length(idx) / length(id));
        % 


    end
end
