%%
% 由机器不会学习翻译
% CSDN：机器不会学习CSJ
% 面包多链接： https://mbd.pub/o/curry/work
%%
function [Best_score, Best_pos, curve] = PO(N, Max_iter, lb, ub, dim, fobj)

% BestF：某个迭代中的最佳值
% WorstF： 某次迭代中的最差值
% GBestF：全局最佳适配值
% AveF：每次迭代的平均值

if (max(size(ub)) == 1)
    ub = ub .* ones(1, dim);
    lb = lb .* ones(1, dim);
end

%% %% 初始化
X0 = initialization(N, dim, ub, lb);  
X = X0;

%% 计算初始适应度值
fitness = zeros(1, N);
for i = 1:N
    fitness(i) = fobj(X(i, :));
end

[fitness, index] = sort(fitness); % 排序
GBestF = fitness(1); %  全局最优
GBestF
AveF = mean(fitness);
for i = 1:N
    X(i, :) = X0(index(i), :);
end
curve = zeros(1, Max_iter);
avg_fitness_curve = zeros(1, Max_iter);
GBestX = X(1, :); %  获取全局最优位置
X_new = X;
%% 设置记录适应度值的数组
search_history = zeros(N, Max_iter, dim);
fitness_history = zeros(N, Max_iter);

%% 开始迭代
for i = 1:Max_iter
    if mod(i,100) == 0
      display(['At iteration ', num2str(i), ' the fitness is ', num2str(curve(i-1))]);
    end
    avg_fitness_curve(i) = AveF;
    alpha = rand(1) / 5;
    sita = rand(1) * pi;
    for j = 1:size(X, 1)
        St = randi([1, 4]);  
        %% 对应论文中的四个步骤
        %% 觅食行为
        if St == 1
                X_new(j, :) = (X(j, :) - GBestX) .* Levy(dim) + rand(1) * mean(X(j, :)) * (1 - i / Max_iter) ^ (2 * i / Max_iter);

        %% 停留行为
        elseif St == 2
                X_new(j, :) = X(j, :) + GBestX .* Levy(dim) + randn() * (1 - i / Max_iter) * ones(1, dim);

        %% 交流行为
        elseif St == 3
                H = rand(1);
                if H < 0.5
                    X_new(j, :) = X(j, :) + alpha * (1 - i / Max_iter) * (X(j, :) - mean(X(j, :)));
                else
                    X_new(j, :) = X(j, :) + alpha * (1 - i / Max_iter) * exp(-j / (rand(1) * Max_iter));
                end
        %% 害怕陌生人的行为
        else
                X_new(j, :) = X(j, :) + rand() * cos((pi *i )/ (2 * Max_iter)) * (GBestX - X(j, :)) - cos(sita) * (i / Max_iter) ^ (2 / Max_iter) * (X(j, :) - GBestX);
        end

         %% 边界判断
        for j = 1:N
            for a = 1:dim
                if (X_new(j, a) > ub(a))
                    X_new(j, a) = ub(a);
                end
                if (X_new(j, a) < lb(a))
                    X_new(j, a) = lb(a);
                end
            end
        end

        %% 更新位置
        for j = 1:N
            fitness_new(j) = fobj(X_new(j, :));
        end
        for j = 1:N
            if (fitness_new(j) < GBestF)
                GBestF = fitness_new(j);
                GBestX = X_new(j, :);
            end
        end
        X = X_new;
        fitness = fitness_new;
        
        %% 排序和更新适应度
        [fitness, index] = sort(fitness); % sort
        for j = 1:N
            X(j, :) = X(index(j), :);
        end
        curve(i) = GBestF;

        
    end
    %% 迭代结束 获取返回参数
    Best_pos = GBestX;
    Best_score = curve(end);
    search_history(:, i, :) = X;
    fitness_history(:, i) = fitness;
    disp(['current iteration is: ',num2str(i), ', best fitness is: ', num2str(GBestF)]);
end

%%  Levy 搜索策略
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1 + beta) *sin(pi * beta / 2) / (gamma((1 + beta) / 2) * beta * 2^((beta - 1) / 2)))^(1 / beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v).^(1 / beta);
    o = step;
end   

end