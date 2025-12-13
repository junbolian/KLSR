function [BestScore, BestPos, BestCost] = DE_KLSR(nPop, MaxIt, VarMin, VarMax, nVar, CostFunction, kopt)
% DE + KLSR（核等值面反射）
% —— 仅两参可调：kopt.tau（拟合周期 τ）与 kopt.p0（初始触发概率）

    VarSize = [1 nVar]; F = 0.5; pCR = 0.9;

    % 初始化
    BestSol.Cost = inf;
    pop = repmat(struct('Position',[],'Cost',inf), nPop, 1);
    for i = 1:nPop
        pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
        pop(i).Cost     = CostFunction(pop(i).Position);
        if pop(i).Cost < BestSol.Cost, BestSol = pop(i); end
    end
    BestCost = zeros(MaxIt,1);

    % ===== 两参 + 默认（若未传 kopt 则走默认） =====
    if nargin < 7 || isempty(kopt), kopt = struct; end
    if ~isfield(kopt,'tau'), kopt.tau = 50; end
    if ~isfield(kopt,'p0'),  kopt.p0  = 0.30; end
    if ~isfield(kopt,'p1'),  kopt.p1  = 0.10; end
    if ~isfield(kopt,'stall_T'),        kopt.stall_T = 7; end
    if ~isfield(kopt,'only_on_stall'),  kopt.only_on_stall = true; end
    if ~isfield(kopt,'quota'),          kopt.quota = 0.10; end
    if ~isfield(kopt,'bound_mode'),     kopt.bound_mode = 'clip'; end
    if ~isfield(kopt,'alpha'),          kopt.alpha = [1 0.5 0.25]; end
    if ~isfield(kopt,'max_evals'),      kopt.max_evals = 1; end
    if ~isfield(kopt,'fit'),            kopt.fit = struct('D',128,'sigma',1.0,'lambda',1e-3, ...
                                           'k',min(3*nVar,200),'df',inf,'quality_min',0.6); end
    if ~isfield(kopt,'arch_max'),       kopt.arch_max = 500; end
    if ~isfield(kopt,'warmup'),         kopt.warmup   = max(100, round(0.02*MaxIt)); end
    if ~isfield(kopt,'stallG'),         kopt.stallG   = 20; end

    p_start = kopt.p0; p_end = kopt.p1;
    fit_period = kopt.tau; fit_opts = kopt.fit;
    alpha_list = kopt.alpha; max_evals = kopt.max_evals; bound_mode = kopt.bound_mode;
    arch_max = kopt.arch_max; warmup = kopt.warmup; stallG = kopt.stallG;

    last_improve = zeros(nPop,1);
    last_gbest_improve = 1;

    % 档案（滑窗）
    arch_X = zeros(0,nVar); arch_F = zeros(0,1);
    model = struct('has',false);

    for it = 1:MaxIt
        % 全局镜面拟合：预热 + 全局停滞 + 低频
        if it >= warmup && (it - last_gbest_improve) >= stallG ...
           && mod(it,fit_period)==1 && size(arch_X,1) >= max(20,nVar+5)
            archive = struct('X',arch_X,'F',arch_F);
            model = KLSR_fitModel(archive, BestSol.Position, VarMin, VarMax, fit_opts);
        end

        % 触发概率
        p_trig = p_start + (it-1)/(MaxIt-1)*(p_end - p_start);

        % 配额：只挑最停滞的 10%
        quota = max(3, ceil(kopt.quota*nPop));
        [~,ord] = sort(it - last_improve,'descend');
        idx_try = ord(1:quota);

        for i = 1:nPop
            x = pop(i).Position;

            % 变异（DE/rand/1）
            A = randperm(nPop); A(A==i)=[];
            a = A(1); b = A(2); c = A(3);
            y = pop(a).Position + F*(pop(b).Position - pop(c).Position);
            y = min(max(y, VarMin), VarMax);

            % 交叉
            z = x; j0 = randi([1 numel(x)]);
            for j = 1:numel(x)
                if j == j0 || rand <= pCR, z(j) = y(j); end
            end

            cand_pos  = z;
            cand_cost = CostFunction(cand_pos);

            % 档案
            if size(arch_X,1) < arch_max
                arch_X(end+1,:) = cand_pos; arch_F(end+1,1) = cand_cost;
            else
                arch_X = [arch_X(2:end,:); cand_pos]; arch_F = [arch_F(2:end); cand_cost];
            end

            % KLSR（择优保底），仅在配额内个体上尝试
            do_it = (rand < p_trig) && ismember(i, idx_try);
            if do_it
                opts = struct('fx',cand_cost,'alpha',alpha_list,'max_evals',max_evals,...
                              'use_pg',true,'bound',bound_mode);
                [cand_pos2, cand_cost2, ~] = KLSR(cand_pos, x, BestSol.Position, VarMin, VarMax, CostFunction, model, opts);
                if cand_cost2 < cand_cost
                    cand_pos  = cand_pos2; cand_cost = cand_cost2;
                    if size(arch_X,1) < arch_max
                        arch_X(end+1,:) = cand_pos; arch_F(end+1,1) = cand_cost;
                    else
                        arch_X = [arch_X(2:end,:); cand_pos]; arch_F = [arch_F(2:end); cand_cost];
                    end
                end
            end

            % 选择
            if cand_cost < pop(i).Cost
                pop(i).Position = cand_pos;
                pop(i).Cost     = cand_cost;
                last_improve(i) = it;
                if cand_cost < BestSol.Cost
                    BestSol = pop(i);
                    last_gbest_improve = it;
                end
            end
        end

        BestCost(it) = BestSol.Cost;
    end

    BestScore = BestSol.Cost;
    BestPos   = BestSol.Position;
end
