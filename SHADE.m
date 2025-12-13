function [fitnessBestX, bestX, Convergence_curve] = SHADE(SearchAgents_no, Gmax, lb, ub, dim, fobj)
    if (max(size(ub)) == 1)
        ub = ub .* ones(1, dim);
        lb = lb .* ones(1, dim);
    end

    H = SearchAgents_no; % 记忆容量
    MCR = 0.5 * ones(H, 1);
    MF = 0.5 * ones(H, 1);
    A = []; % 外部存档 淘汰解
    B = []; % 淘汰解 成功解 适应度 算权重

    % 初始化种群
    x = repmat(lb, SearchAgents_no, 1) + rand(SearchAgents_no, dim) .* (repmat(ub - lb, SearchAgents_no, 1));
    fitness = zeros(1, SearchAgents_no);
    for i = 1:SearchAgents_no
        fitness(i) = fobj(x(i, :));
    end

    % 初始化 search_history 和 fitness_history
    search_history = zeros(SearchAgents_no, Gmax, dim);
    fitness_history = zeros(SearchAgents_no, Gmax);

    FES = SearchAgents_no;
    q = 1; % 当前单元
    G = 0;
    % 主迭代
    while G < Gmax
        [~, I] = sort(fitness);
        SCR = [];
        SF = [];
        w = [];
        B = [];
        pmin = 2 / SearchAgents_no;
        for i = 1:SearchAgents_no
            r = randi(H); % 随机选一个存档
            CR(i) = normrnd(MCR(r), 0.1);
            CR(i) = max(0, CR(i));
            CR(i) = min(1, CR(i));
            F(i) = cauchyrnd(MF(r), 0.1);
            F(i) = min(1, F(i));
            while (F(i) <= 0)
                F(i) = cauchyrnd(MF(r), 0.1);
                F(i) = min(1, F(i));
            end
            p = unifrnd(pmin, 0.2); % 随机生成参考个体率
            p0 = round(p * SearchAgents_no);
            temp1 = floor(rand * p0) + 1; % 选择一个参考个体作最优引导
            % 随机选两个个体
            r1 = floor(rand * SearchAgents_no) + 1;
            while r1 == i
                r1 = floor(rand * SearchAgents_no) + 1;
            end
            S = [x; A];
            r = size(S, 1);
            r2 = floor(rand * r) + 1;
            while r2 == i || r2 == r1
                r2 = floor(rand * r) + 1;
            end
            % 变异
            v(i, :) = x(i, :) + F(i) * (x(I(temp1), :) - x(i, :)) + F(i) * (x(r1, :) - S(r2, :));
            jrand = floor(rand * dim) + 1;
            for j = 1:dim
                % 范围规约
                if v(i, j) > ub(j)
                    v(i, j) = max(lb(j), 2 * ub(j) - v(i, j));
                end
                if v(i, j) < lb(j)
                    v(i, j) = min(ub(j), 2 * lb(j) - v(i, j));
                end
                % 交叉
                if (rand < CR(i)) || (j == jrand)
                    newx(i, j) = v(i, j);
                else
                    newx(i, j) = x(i, j);
                end
            end
            fnew(i) = fobj(newx(i, :));
            % 有淘汰解 更新外部存档
            if fnew(i) < fitness(i)
                A = [A; x(i, :)];
                B = [B; fnew(i) fitness(i)];
                SCR = [SCR; CR(i)];
                SF = [SF; F(i)];
                x(i, :) = newx(i, :);
                fitness(i) = fnew(i);
            end
        end
        a = size(A, 1);
        % 存档超出容量 删除
        while a > SearchAgents_no
            temp3 = floor(rand(a - SearchAgents_no, 1) * a) + 1;
            A(temp3, :) = [];
            a = size(A, 1);
        end

        % 更新历史记忆
        if isempty(SCR) && isempty(SF)
            ;
        else
            % 算权重
            sumf = 0;
            b = abs(B(:, 2) - B(:, 1));
            sumf = sum(b);
            w = b ./ sumf;
            MCR(q) = sum(w .* SCR);
            MF(q) = sum(w .* (SF .^ 2)) / sum(w .* SF);
            q = q + 1;
            if q > H
                q = 1;
            end
        end

        % 更新 search_history 和 fitness_history
        search_history(:, G + 1, :) = x;
        fitness_history(:, G + 1) = fitness;

        G = G + 1;
        [fitnessBestX, indx] = min(fitness);
        bestX = x(indx, :);
        Convergence_curve(G) = fitnessBestX;
    end
end
