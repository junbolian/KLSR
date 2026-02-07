function [gBestScore, gBest, cg_curve] = GA_KLSR(N, Max_iteration, lb, ub, dim, fobj, kopt)
% GA + KLSR

    % ===== bounds =====
    if isscalar(lb), lb = repmat(lb, 1, dim); else, lb = lb(:)'; end
    if isscalar(ub), ub = repmat(ub, 1, dim); else, ub = ub(:)'; end

    popSize = N;
    iter = Max_iteration;

    % ===== GA defaults =====
    pc = 0.9; pm = 1/dim;
    eta_c = 15; eta_m = 20;
    tourn_k = 2; elite_n = 1;

    % ===== kopt defaults =====
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
                                           'k',min(3*dim,200),'df',inf,'quality_min',0.6); end
    if ~isfield(kopt,'arch_max'),       kopt.arch_max = 500; end
    if ~isfield(kopt,'warmup'),         kopt.warmup   = max(100, round(0.02*iter)); end
    if ~isfield(kopt,'stallG'),         kopt.stallG   = 20; end

    % unpack
    fit_period = kopt.tau; p_start = kopt.p0; p_end = kopt.p1;
    stall_T = kopt.stall_T; only_on_stall = kopt.only_on_stall;
    alpha_list = kopt.alpha; max_evals_default = kopt.max_evals;
    bound_mode = kopt.bound_mode; fit_opts = kopt.fit;
    arch_max = kopt.arch_max; warmup = kopt.warmup; stallG = kopt.stallG;

    max_nfes = popSize + iter * popSize;
    nfes = 0;

    % ===== init =====
    pop = initialization(popSize, dim, ub, lb);
    fit = inf(popSize,1);

    for i = 1:popSize
        pop(i,:) = max(min(pop(i,:), ub), lb);
        fit(i) = fobj(pop(i,:)); 
        nfes = nfes + 1;
    end

    [gBestScore, idx] = min(fit);
    gBest = pop(idx,:);
    cg_curve = nan(1, iter);

    % pBest & Memories
    pBest = pop; pBestScore = fit;
    last_improve = ones(popSize,1);
    last_gbest_improve = 1;

    % Archive
    arch_X = pop; arch_F = fit;
    if size(arch_X,1) > arch_max
        arch_X = arch_X(end-arch_max+1:end,:);
        arch_F = arch_F(end-arch_max+1:end,:);
    end

    model = struct('has', false);

    for gen = 1:iter
        if nfes >= max_nfes
            cg_curve(gen:end) = gBestScore; break;
        end
        
        stopFlag = false;

        % ===== fit mirror model =====
        if gen >= warmup && (gen - last_gbest_improve) >= stallG ...
           && mod(gen, fit_period) == 1 && exist('KLSR_fitModel','file') == 2 ...
           && size(arch_X,1) >= max(20, dim+5)
            archive = struct('X', arch_X, 'F', arch_F);
            model = KLSR_fitModel(archive, gBest, lb, ub, fit_opts);
        end

        % ===== KLSR injection =====
        if iter > 1, p_trig = p_start + (gen-1)/(iter-1) * (p_end - p_start);
        else, p_trig = p_end; end

        quota = max(3, ceil(kopt.quota * popSize));
        [~, ordStall] = sort(gen - last_improve, 'descend');
        idx_try = ordStall(1:min(quota, popSize));

        for t = 1:numel(idx_try)
            i = idx_try(t);
            if nfes >= max_nfes, stopFlag = true; break; end

            do_it = (rand < p_trig) && (~only_on_stall || (gen - last_improve(i) >= stall_T));
            if ~do_it, continue; end

            remain = max_nfes - nfes;
            if remain <= 0, stopFlag = true; break; end

            % [Safe Guard] Dynamic max_evals
            opts = struct('fx', fit(i), 'alpha', alpha_list, ...
                          'max_evals', min(max_evals_default, remain), ...
                          'use_pg', true, 'bound', bound_mode);

            [x2, f2, extra_evals] = KLSR(pop(i,:), pBest(i,:), gBest, lb, ub, fobj, model, opts);
            nfes = nfes + extra_evals;

            if f2 < fit(i)
                pop(i,:) = x2; fit(i) = f2;
                
                % Update Archive
                arch_X = [arch_X; x2]; arch_F = [arch_F; f2]; %#ok<AGROW>
                if size(arch_X,1) > arch_max
                    arch_X = arch_X(end-arch_max+1:end,:);
                    arch_F = arch_F(end-arch_max+1:end,:);
                end
                
                % Update pBest
                if f2 < pBestScore(i)
                    pBestScore(i) = f2; pBest(i,:) = x2; last_improve(i) = gen;
                end
                if f2 < gBestScore
                    gBestScore = f2; gBest = x2; last_gbest_improve = gen;
                end
            end
        end

        if stopFlag
            cg_curve(gen:end) = gBestScore; break;
        end

        % ===== elitism =====
        [fit_sorted, ord] = sort(fit, 'ascend');
        elites = pop(ord(1:elite_n), :);
        elites_fit = fit_sorted(1:elite_n);

        % ===== offspring generation =====
        offspring = zeros(popSize, dim);
        off_fit = inf(popSize,1);

        c = 1;
        while c <= popSize
            p1 = pop(tournament_pick(fit, tourn_k), :);
            p2 = pop(tournament_pick(fit, tourn_k), :);

            if rand < pc
                [ch1, ch2] = sbx_crossover(p1, p2, lb, ub, eta_c);
            else
                ch1 = p1; ch2 = p2;
            end

            ch1 = poly_mutation(ch1, lb, ub, pm, eta_m);
            ch2 = poly_mutation(ch2, lb, ub, pm, eta_m);

            ch1 = max(min(ch1, ub), lb);
            ch2 = max(min(ch2, ub), lb);

            offspring(c,:) = ch1;
            if c+1 <= popSize, offspring(c+1,:) = ch2; end
            c = c + 2;
        end

        % ===== evaluate offspring =====
        for i = 1:popSize
            if nfes >= max_nfes, stopFlag = true; break; end
            off_fit(i) = fobj(offspring(i,:));
            nfes = nfes + 1;
        end

        % [Critical Fix] If budget exhausted, update gBest with partial results
        if stopFlag
            [best_off, idx_off] = min(off_fit);
            if best_off < gBestScore
                gBestScore = best_off;
            end
            cg_curve(gen:end) = gBestScore;
            break;
        end

        % ===== next generation =====
        pop = offspring; fit = off_fit;

        [fit_sorted, ord] = sort(fit, 'ascend');
        worst = ord(end-elite_n+1:end);
        pop(worst,:) = elites;
        fit(worst) = elites_fit;

        % update memories
        for i = 1:popSize
            if fit(i) < pBestScore(i)
                pBestScore(i) = fit(i); pBest(i,:) = pop(i,:); last_improve(i) = gen;
            end
            arch_X = [arch_X; pop(i,:)]; arch_F = [arch_F; fit(i)]; %#ok<AGROW>
            if size(arch_X,1) > arch_max
                arch_X = arch_X(end-arch_max+1:end,:);
                arch_F = arch_F(end-arch_max+1:end,:);
            end
        end

        [genBest, idx] = min(fit);
        if genBest < gBestScore
            gBestScore = genBest; gBest = pop(idx,:); last_gbest_improve = gen;
        end
        cg_curve(gen) = gBestScore;
    end
end

% ===== helpers (Keep unchanged) =====
function idx = tournament_pick(fit, k)
    n = numel(fit);
    cand = randi(n, [k,1]);
    [~, j] = min(fit(cand));
    idx = cand(j);
end
function [c1, c2] = sbx_crossover(p1, p2, lb, ub, eta_c)
    d = numel(p1);
    c1 = p1; c2 = p2;
    for j = 1:d
        if rand <= 0.5
            if abs(p1(j) - p2(j)) > 1e-14
                x1 = min(p1(j), p2(j));
                x2 = max(p1(j), p2(j));
                yl = lb(j); yu = ub(j);
                randu = rand;
                beta = 1 + (2*(x1-yl)/(x2-x1));
                alpha = 2 - beta^(-(eta_c+1));
                if randu <= 1/alpha, betaq = (randu*alpha)^(1/(eta_c+1));
                else, betaq = (1/(2 - randu*alpha))^(1/(eta_c+1)); end
                child1 = 0.5*((x1+x2) - betaq*(x2-x1));
                beta = 1 + (2*(yu-x2)/(x2-x1));
                alpha = 2 - beta^(-(eta_c+1));
                if randu <= 1/alpha, betaq = (randu*alpha)^(1/(eta_c+1));
                else, betaq = (1/(2 - randu*alpha))^(1/(eta_c+1)); end
                child2 = 0.5*((x1+x2) + betaq*(x2-x1));
                child1 = min(max(child1, yl), yu);
                child2 = min(max(child2, yl), yu);
                if rand <= 0.5, c1(j) = child2; c2(j) = child1;
                else, c1(j) = child1; c2(j) = child2; end
            else, c1(j) = p1(j); c2(j) = p2(j); end
        else, c1(j) = p1(j); c2(j) = p2(j); end
    end
end
function y = poly_mutation(x, lb, ub, pm, eta_m)
    y = x;
    d = numel(x);
    for j = 1:d
        if rand < pm
            yl = lb(j); yu = ub(j);
            if yl == yu, continue; end
            delta1 = (y(j) - yl) / (yu - yl);
            delta2 = (yu - y(j)) / (yu - yl);
            r = rand;
            mut_pow = 1/(eta_m+1);
            if r <= 0.5
                xy = 1 - delta1;
                val = 2*r + (1 - 2*r)*(xy^(eta_m+1));
                deltaq = val^mut_pow - 1;
            else
                xy = 1 - delta2;
                val = 2*(1-r) + 2*(r-0.5)*(xy^(eta_m+1));
                deltaq = 1 - val^mut_pow;
            end
            y(j) = y(j) + deltaq*(yu - yl);
            y(j) = min(max(y(j), yl), yu);
        end
    end
end