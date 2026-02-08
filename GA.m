function [gBestScore, gBest, cg_curve] = GA(N, Max_iteration, lb, ub, dim, fobj)
% GA - Real-coded Genetic Algorithm

    % ===== bounds =====
    if isscalar(lb), lb = repmat(lb, 1, dim); else, lb = lb(:)'; end
    if isscalar(ub), ub = repmat(ub, 1, dim); else, ub = ub(:)'; end

    popSize = N;
    iter = Max_iteration;

    % ===== GA defaults =====
    pc = 0.9; pm = 1/dim;
    eta_c = 15; eta_m = 20;
    tourn_k = 2; elite_n = 1;

    % ===== budget =====
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

    % ===== main loop =====
    for gen = 1:iter
        if nfes >= max_nfes
            cg_curve(gen:end) = gBestScore; break;
        end

        % --- elitism ---
        [fit_sorted, ord] = sort(fit, 'ascend');
        elites = pop(ord(1:elite_n), :);
        elites_fit = fit_sorted(1:elite_n);

        % --- create offspring ---
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

        % evaluate offspring with budget guard
        stopFlag = false;
        for i = 1:popSize
            if nfes >= max_nfes
                stopFlag = true; break;
            end
            off_fit(i) = fobj(offspring(i,:)); 
            nfes = nfes + 1;
        end

        if stopFlag
            [best_off, idx_off] = min(off_fit);
            if best_off < gBestScore
                gBestScore = best_off;
                gBest = offspring(idx_off,:);
            end
            cg_curve(gen:end) = gBestScore;
            break;
        end

        % --- next generation ---
        pop = offspring;
        fit = off_fit;

        % inject elites
        [fit_sorted, ord] = sort(fit, 'ascend');
        worst = ord(end-elite_n+1:end);
        pop(worst,:) = elites;
        fit(worst) = elites_fit;

        % update global best
        [genBest, idx] = min(fit);
        if genBest < gBestScore
            gBestScore = genBest;
            gBest = pop(idx,:);
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
            else
                c1(j) = p1(j); c2(j) = p2(j);
            end
        end
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