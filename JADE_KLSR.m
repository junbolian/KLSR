function [fitnessBestP, bestP, Convergence_curve] = JADE_KLSR(SearchAgents_no, Gmax, lb, ub, dim, fobj, kopt)
% JADE + KLSR
% - 在 JADE 的 trial 向量 U 评估后，额外做一次 KLSR（有预算/概率/配额/停滞门控）
% - KLSR 的镜面模型按 tau 代周期拟合（warmup + stallG + 档案足够时）
%
% 输出：
%   fitnessBestP: 全局最优值
%   bestP       : 全局最优解
%   Convergence_curve(G): 每代的 best 值

    % ------------------- Bounds normalize -------------------
    if (max(size(ub)) == 1)
        ub = ub .* ones(1, dim);
        lb = lb .* ones(1, dim);
    end
    lb = lb(:)'; ub = ub(:)';

    % ------------------- JADE params -------------------
    c = 1/10;           % JADE 参数更新系数
    p = 0.05;           % p-best 比例
    top = max(1, floor(p * SearchAgents_no));

    % JADE 的 archive（原算法）
    A = [];
    tA = 0;

    % 初始均值
    uCR = 0.5;
    uF  = 0.5;

    % ------------------- KLSR defaults -------------------
    if nargin < 7 || isempty(kopt), kopt = struct; end
    if ~isfield(kopt,'tau'), kopt.tau = 50; end          % 拟合周期
    if ~isfield(kopt,'p0'),  kopt.p0  = 0.30; end        % 初始触发概率
    if ~isfield(kopt,'p1'),  kopt.p1  = 0.10; end        % 末期触发概率
    if ~isfield(kopt,'stall_T'),        kopt.stall_T = 7; end
    if ~isfield(kopt,'only_on_stall'),  kopt.only_on_stall = true; end
    if ~isfield(kopt,'quota'),          kopt.quota = 0.10; end
    if ~isfield(kopt,'bound_mode'),     kopt.bound_mode = 'clip'; end
    if ~isfield(kopt,'alpha'),          kopt.alpha = [1 0.5 0.25]; end
    if ~isfield(kopt,'max_evals'),      kopt.max_evals = 1; end
    if ~isfield(kopt,'fit'),            kopt.fit = struct('D',128,'sigma',1.0,'lambda',1e-3, ...
                                           'k',min(3*dim,200),'df',inf,'quality_min',0.6); end
    if ~isfield(kopt,'arch_max'),       kopt.arch_max = 500; end
    if ~isfield(kopt,'warmup'),         kopt.warmup   = max(100, round(0.02*Gmax)); end
    if ~isfield(kopt,'stallG'),         kopt.stallG   = 20; end

    p_start = kopt.p0;  p_end = kopt.p1;
    fit_period = kopt.tau; fit_opts = kopt.fit;
    alpha_list = kopt.alpha; max_evals = kopt.max_evals; bound_mode = kopt.bound_mode;
    arch_max = kopt.arch_max; warmup = kopt.warmup; stallG = kopt.stallG;
    stall_T = kopt.stall_T; only_on_stall = kopt.only_on_stall;

    % ------------------- init population -------------------
    P = repmat(lb, SearchAgents_no, 1) + rand(SearchAgents_no, dim) .* repmat((ub - lb), SearchAgents_no, 1);

    fitnessP = zeros(1, SearchAgents_no);
    for i = 1:SearchAgents_no
        fitnessP(i) = fobj(P(i,:));
    end

    % 全局 best
    [fitnessBestP, indexBestP] = min(fitnessP);
    bestP = P(indexBestP, :);

    % 个体历史最优（给 KLSR 的 p 用）
    pBest = P;
    pBestScore = fitnessP;

    % KLSR 的停滞追踪
    last_improve = zeros(SearchAgents_no, 1);  % 记录每个个体最近一次"被接受更新"的代数
    last_gbest_improve = 1;

    % KLSR 的档案（滑窗）
    arch_X = P;               % 用初始种群热启动
    arch_F = fitnessP(:);
    if size(arch_X,1) > arch_max
        arch_X = arch_X(end-arch_max+1:end,:);
        arch_F = arch_F(end-arch_max+1:end,:);
    end
    model = struct('has', false);

    % 收敛曲线
    Convergence_curve = zeros(1, Gmax);

    % ------------------- main loop -------------------
    G = 0;
    while G < Gmax
        G = G + 1;

        % ========= (1) 拟合镜面：warmup + 全局停滞 + 低频 =========
        if G >= warmup && (G - last_gbest_improve) >= stallG ...
           && mod(G, fit_period) == 1 && size(arch_X,1) >= max(20, dim+5)
            archive = struct('X', arch_X, 'F', arch_F);
            model = KLSR_fitModel(archive, bestP, lb, ub, fit_opts);
        end

        % ========= (2) 本代触发概率 & 配额（只给最停滞的那部分） =========
        if Gmax > 1
            p_trig = p_start + (G-1)/(Gmax-1) * (p_end - p_start);
        else
            p_trig = p_start;
        end

        quota = max(3, ceil(kopt.quota * SearchAgents_no));
        [~, ord] = sort(G - last_improve, 'descend');
        idx_try = ord(1:quota);

        % ========= (3) 采样 CR / F =========
        CR = zeros(1, SearchAgents_no);
        Fv = zeros(1, SearchAgents_no);

        for i = 1:SearchAgents_no
            % normrnd(uCR,0.1) -> uCR + 0.1*randn (不依赖 stats toolbox)
            CR(i) = uCR + 0.1*randn;
            while (CR(i) > 1 || CR(i) < 0)
                CR(i) = uCR + 0.1*randn;
            end

            Fi = cauchyrnd(uF, 0.1);
            while Fi <= 0
                Fi = cauchyrnd(uF, 0.1);
            end
            if Fi > 1, Fi = 1; end
            Fv(i) = Fi;
        end

        % ========= (4) 取 top p-best 集合 =========
        [~, indexSortP] = sort(fitnessP);
        bestTopP = P(indexSortP(1:top), :);

        % ========= (5) 变异 + 交叉 -> 产生 U =========
        V = zeros(SearchAgents_no, dim);
        U = zeros(SearchAgents_no, dim);

        for i = 1:SearchAgents_no
            % Xpbest 从 top 随机选
            k0 = randi(top);
            Xpbest = bestTopP(k0, :);

            % P1 从 P 随机选（不等于 i）
            k1 = randi(SearchAgents_no);
            while k1 == i
                k1 = randi(SearchAgents_no);
            end
            P1 = P(k1, :);

            % P2 从 P ∪ A 随机选（不等于 i 和 k1）
            PandA = [P; A];
            num = size(PandA, 1);
            k2 = randi(num);
            P2 = PandA(k2, :);
            % 简单避免重复（够用就行）
            while (k2 == i) || (k2 == k1)
                k2 = randi(num);
                P2 = PandA(k2, :);
            end

            % DE/current-to-pbest/1
            V(i,:) = P(i,:) + Fv(i).*(Xpbest - P(i,:)) + Fv(i).*(P1 - P2);
        end

        % 交叉
        for i = 1:SearchAgents_no
            jrand = randi([1, dim]);
            for j = 1:dim
                if (rand <= CR(i)) || (j == jrand)
                    U(i,j) = V(i,j);
                else
                    U(i,j) = P(i,j);
                end
            end
        end

        % ========= (6) 边界处理（保持你原来的"越界就随机拉回"方式） =========
        for i = 1:SearchAgents_no
            for j = 1:dim
                while (U(i,j) > ub(j) || U(i,j) < lb(j))
                    U(i,j) = (ub(j) - lb(j)) * rand + lb(j);
                end
            end
        end

        % ========= (7) 选择 + (可选) KLSR =========
        Scr = []; Sf = [];
        for i = 1:SearchAgents_no
            % trial 先评估一次
            fu = fobj(U(i,:));

            % 档案入库（trial）
            arch_X = [arch_X; U(i,:)];
            arch_F = [arch_F; fu];
            if size(arch_X,1) > arch_max
                arch_X = arch_X(end-arch_max+1:end,:);
                arch_F = arch_F(end-arch_max+1:end,:);
            end

            % --- KLSR 触发：概率 + 配额 + (可选) 停滞门控 ---
            do_it = (rand < p_trig) && ismember(i, idx_try) ...
                    && (~only_on_stall || (G - last_improve(i) >= stall_T));

            if do_it
                opts = struct('fx', fu, 'alpha', alpha_list, 'max_evals', max_evals, ...
                              'use_pg', true, 'bound', bound_mode);

                % 这里 p 用"个体历史最优"，g 用"全局最优"
                [u2, fu2, ~] = KLSR(U(i,:), pBest(i,:), bestP, lb, ub, fobj, model, opts);

                if fu2 < fu
                    U(i,:) = u2;
                    fu = fu2;

                    % 改进后的 trial 也入档案
                    arch_X = [arch_X; U(i,:)];
                    arch_F = [arch_F; fu];
                    if size(arch_X,1) > arch_max
                        arch_X = arch_X(end-arch_max+1:end,:);
                        arch_F = arch_F(end-arch_max+1:end,:);
                    end
                end
            end

            % --- JADE selection ---
            if fu < fitnessP(i)
                % 被淘汰的父代进 A
                A = [A; P(i,:)];
                tA = tA + 1;

                % 更新到新个体
                P(i,:) = U(i,:);
                fitnessP(i) = fu;

                % 记录成功参数
                Scr(end+1) = CR(i); %#ok<AGROW>
                Sf(end+1)  = Fv(i); %#ok<AGROW>

                % 停滞更新
                last_improve(i) = G;

                % 个体历史最优
                if fu < pBestScore(i)
                    pBestScore(i) = fu;
                    pBest(i,:) = P(i,:);
                end

                % 全局最优
                if fu < fitnessBestP
                    fitnessBestP = fu;
                    bestP = P(i,:);
                    last_gbest_improve = G;
                end
            end
        end

        % ========= (8) 控制 A 的规模 <= NP =========
        if size(A,1) > SearchAgents_no
            nRem = size(A,1) - SearchAgents_no;
            k4 = randperm(size(A,1), nRem);
            A(k4,:) = [];
            tA = size(A,1);
        end

        % ========= (9) 更新 uCR / uF =========
        if ~isempty(Scr)
            newSf = (sum(Sf.^2)) / (sum(Sf) + eps);
            uCR = (1-c)*uCR + c*mean(Scr);
            uF  = (1-c)*uF  + c*newSf;
        end

        Convergence_curve(G) = fitnessBestP;
    end
end
