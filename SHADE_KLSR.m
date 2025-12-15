function [fitnessBestX, bestX, Convergence_curve] = SHADE_KLSR(SearchAgents_no, Gmax, lb, ub, dim, fobj, kopt)
% SHADE + KLSR
% - 基于你提供的 SHADE.m 结构
% - 在 trial 向量 newx 评估后，按概率/配额/停滞门控调用一次 KLSR
% - KLSR 镜面模型按 tau 周期拟合（warmup + stallG + 档案足够时）
%
% 依赖：
%   cauchyrnd.m
%   KLSR.m, KLSR_fitModel.m

    % bounds normalize
    if (max(size(ub)) == 1)
        ub = ub .* ones(1, dim);
        lb = lb .* ones(1, dim);
    end
    lb = lb(:)'; ub = ub(:)';

    % KLSR defaults
    if nargin < 7 || isempty(kopt), kopt = struct; end
    if ~isfield(kopt,'tau'), kopt.tau = 50; end
    if ~isfield(kopt,'p0'),  kopt.p0  = 0.30; end
    if ~isfield(kopt,'p1'),  kopt.p1  = 0.10; end
    if ~isfield(kopt,'stall_T'),       kopt.stall_T = 7; end
    if ~isfield(kopt,'only_on_stall'), kopt.only_on_stall = true; end
    if ~isfield(kopt,'quota'),         kopt.quota = 0.10; end
    if ~isfield(kopt,'bound_mode'),    kopt.bound_mode = 'clip'; end
    if ~isfield(kopt,'alpha'),         kopt.alpha = [1 0.5 0.25]; end
    if ~isfield(kopt,'max_evals'),     kopt.max_evals = 1; end
    if ~isfield(kopt,'fit'),           kopt.fit = struct('D',128,'sigma',1.0,'lambda',1e-3, ...
                                          'k',min(3*dim,200),'df',inf,'quality_min',0.6); end
    if ~isfield(kopt,'arch_max'),      kopt.arch_max = 500; end
    if ~isfield(kopt,'warmup'),        kopt.warmup = max(100, round(0.02*Gmax)); end
    if ~isfield(kopt,'stallG'),        kopt.stallG = 20; end

    p_start      = kopt.p0;
    p_end        = kopt.p1;
    fit_period   = kopt.tau;
    fit_opts     = kopt.fit;
    alpha_list   = kopt.alpha;
    max_evals    = kopt.max_evals;
    bound_mode   = kopt.bound_mode;
    arch_max     = kopt.arch_max;
    warmup       = kopt.warmup;
    stallG       = kopt.stallG;
    stall_T      = kopt.stall_T;
    only_on_stall= kopt.only_on_stall;

    % SHADE params
    H   = SearchAgents_no;           % memory size
    MCR = 0.5 * ones(H, 1);
    MF  = 0.5 * ones(H, 1);
    q   = 1;                         % memory index

    Aext = [];                       % external archive (for SHADE mutation)

    % init population
    X = repmat(lb, SearchAgents_no, 1) + rand(SearchAgents_no, dim) .* repmat((ub - lb), SearchAgents_no, 1);

    fitness = zeros(SearchAgents_no,1);
    for i = 1:SearchAgents_no
        fitness(i) = fobj(X(i,:));
    end

    % global best
    [fitnessBestX, bestIdx] = min(fitness);
    bestX = X(bestIdx,:);

    % personal best (for KLSR p-center)
    pBest      = X;
    pBestScore = fitness;

    % stall tracking (for KLSR)
    last_improve      = zeros(SearchAgents_no,1);
    last_gbest_improve= 1;

    % KLSR archive (sliding window of evaluated points)
    arch_X = X;
    arch_F = fitness;
    if size(arch_X,1) > arch_max
        arch_X = arch_X(end-arch_max+1:end,:);
        arch_F = arch_F(end-arch_max+1:end,:);
    end
    model = struct('has', false);

    % convergence
    Convergence_curve = zeros(1, Gmax);

    %main loop
    for G = 1:Gmax

        % ==== (1) fit mirror model (warmup + global stall + low frequency) ====
        if G >= warmup && (G - last_gbest_improve) >= stallG ...
                && mod(G, fit_period) == 1 && size(arch_X,1) >= max(20, dim+5)
            archive = struct('X', arch_X, 'F', arch_F);
            model = KLSR_fitModel(archive, bestX, lb, ub, fit_opts);
        end

        % ==== (2) trigger prob & quota ====
        if Gmax > 1
            p_trig = p_start + (G-1)/(Gmax-1) * (p_end - p_start);
        else
            p_trig = p_start;
        end

        quota = max(3, ceil(kopt.quota * SearchAgents_no));
        [~, ord] = sort(G - last_improve, 'descend');
        idx_try = ord(1:quota);

        % ==== (3) sort for p-best selection ====
        [~, I] = sort(fitness);
        pmin = 2 / SearchAgents_no;

        % success sets (for SHADE memory update)
        SCR = [];
        SF  = [];
        dF  = [];   % improvement magnitude for weights

        % prealloc
        CR = zeros(SearchAgents_no,1);
        Fv = zeros(SearchAgents_no,1);
        V  = zeros(SearchAgents_no, dim);
        U  = zeros(SearchAgents_no, dim);
        fU = inf(SearchAgents_no,1);

        % ==== (4) generate trials ====
        for i = 1:SearchAgents_no

            % sample CR, F from memory (no stats toolbox)
            r = randi(H);
            cr = MCR(r) + 0.1*randn;
            cr = max(0, min(1, cr));
            fi = cauchyrnd(MF(r), 0.1);
            while fi <= 0
                fi = cauchyrnd(MF(r), 0.1);
            end
            if fi > 1, fi = 1; end

            CR(i) = cr;
            Fv(i) = fi;

            % p-best rate and pick one from top p0
            p_rate = pmin + (0.2 - pmin) * rand;   % unifrnd(pmin,0.2)
            p0 = max(1, round(p_rate * SearchAgents_no));
            temp1 = randi(p0);
            Xpbest = X(I(temp1),:);

            % choose r1 != i
            r1 = randi(SearchAgents_no);
            while r1 == i
                r1 = randi(SearchAgents_no);
            end

            % choose r2 from [X; Aext] not equal to i or r1 (roughly)
            S = [X; Aext];
            rS = size(S,1);
            r2 = randi(rS);
            while (r2 == i) || (r2 == r1)
                r2 = randi(rS);
            end

            % mutation (current-to-pbest/1)
            V(i,:) = X(i,:) + fi*(Xpbest - X(i,:)) + fi*(X(r1,:) - S(r2,:));

            % bound handling (mirror like your baseline)
            for j = 1:dim
                if V(i,j) > ub(j)
                    V(i,j) = max(lb(j), 2*ub(j) - V(i,j));
                elseif V(i,j) < lb(j)
                    V(i,j) = min(ub(j), 2*lb(j) - V(i,j));
                end
            end

            % crossover
            jrand = randi(dim);
            for j = 1:dim
                if (rand < cr) || (j == jrand)
                    U(i,j) = V(i,j);
                else
                    U(i,j) = X(i,j);
                end
            end

            % evaluate trial
            fu = fobj(U(i,:));
            fU(i) = fu;

            % push trial to KLSR archive
            arch_X = [arch_X; U(i,:)];
            arch_F = [arch_F; fu];
            if size(arch_X,1) > arch_max
                arch_X = arch_X(end-arch_max+1:end,:);
                arch_F = arch_F(end-arch_max+1:end,:);
            end

            % ==== (5) optional KLSR on trial ====
            do_it = (rand < p_trig) && ismember(i, idx_try) ...
                    && (~only_on_stall || (G - last_improve(i) >= stall_T));

            if do_it
                opts = struct('fx', fu, 'alpha', alpha_list, 'max_evals', max_evals, ...
                              'use_pg', true, 'bound', bound_mode);

                % p uses personal-best, g uses global-best
                [u2, fu2, ~] = KLSR(U(i,:), pBest(i,:), bestX, lb, ub, fobj, model, opts);

                if fu2 < fu
                    U(i,:) = u2;
                    fu = fu2;
                    fU(i) = fu;

                    % push improved trial
                    arch_X = [arch_X; U(i,:)];
                    arch_F = [arch_F; fu];
                    if size(arch_X,1) > arch_max
                        arch_X = arch_X(end-arch_max+1:end,:);
                        arch_F = arch_F(end-arch_max+1:end,:);
                    end
                end
            end

        end

        % ==== (6) selection + update external archive + collect successes ====
        for i = 1:SearchAgents_no
            if fU(i) < fitness(i)
                % external archive stores replaced parent
                Aext = [Aext; X(i,:)];
                if size(Aext,1) > SearchAgents_no
                    % randomly delete overflow
                    nRem = size(Aext,1) - SearchAgents_no;
                    ridx = randperm(size(Aext,1), nRem);
                    Aext(ridx,:) = [];
                end

                % improvement magnitude for weights
                dF(end+1,1) = abs(fitness(i) - fU(i)); %#ok<AGROW>

                % record successful CR/F
                SCR(end+1,1) = CR(i); %#ok<AGROW>
                SF(end+1,1)  = Fv(i); %#ok<AGROW>

                % accept
                X(i,:) = U(i,:);
                fitness(i) = fU(i);

                last_improve(i) = G;

                % update personal best
                if fitness(i) < pBestScore(i)
                    pBestScore(i) = fitness(i);
                    pBest(i,:) = X(i,:);
                end

                % update global best
                if fitness(i) < fitnessBestX
                    fitnessBestX = fitness(i);
                    bestX = X(i,:);
                    last_gbest_improve = G;
                end
            end
        end

        % ==== (7) update SHADE memory ====
        if ~isempty(SCR)
            w = dF / (sum(dF) + eps);
            MCR(q) = sum(w .* SCR);
            MF(q)  = sum(w .* (SF.^2)) / (sum(w .* SF) + eps);

            q = q + 1;
            if q > H, q = 1; end
        end

        Convergence_curve(G) = fitnessBestX;
    end
end
