function [bestScore, bestX, curve] = SHADE_KLSR( ...
    nPop, MaxIt, lb, ub, dim, fobj, kopt)

    %%初始化（直接复用 SHADE 逻辑）
    if isscalar(lb), lb = lb .* ones(1,dim); end
    if isscalar(ub), ub = ub .* ones(1,dim); end

    % SHADE 参数
    H   = nPop;
    MCR = 0.5 * ones(H,1);
    MF  = 0.5 * ones(H,1);
    A   = [];

    % 初始化种群
    X = lb + rand(nPop,dim) .* (ub - lb);
    f = zeros(nPop,1);
    for i = 1:nPop
        f(i) = fobj(X(i,:));
    end

    [bestScore, idx] = min(f);
    bestX = X(idx,:);
    curve = zeros(1, MaxIt);

    %% KLSR 状态
    klsr_state = struct('t',1);

    %%主循环
    for G = 1:MaxIt

        [f_sorted, I] = sort(f);
        SCR = []; SF = []; B = [];

        for i = 1:nPop
            r = randi(H);
            CR = min(1,max(0,normrnd(MCR(r),0.1)));
            F  = cauchyrnd(MF(r),0.1);
            while F <= 0, F = cauchyrnd(MF(r),0.1); end
            F = min(F,1);

            pmin = 2/nPop;
            p = unifrnd(pmin,0.2);
            pbest = I(randi(round(p*nPop)));

            r1 = randi(nPop); while r1==i, r1=randi(nPop); end
            S = [X; A];
            r2 = randi(size(S,1));

            v = X(i,:) + F*(X(pbest,:)-X(i,:)) + F*(X(r1,:)-S(r2,:));
            v = min(max(v,lb),ub);

            jrand = randi(dim);
            u = X(i,:);
            for j = 1:dim
                if rand < CR || j==jrand
                    u(j) = v(j);
                end
            end

            fu = fobj(u);

            if fu < f(i)
                A = [A; X(i,:)];
                SCR = [SCR; CR];
                SF  = [SF; F];
                B   = [B; f(i)-fu];
                X(i,:) = u;
                f(i) = fu;
            end
        end

        % 更新 MCR / MF
        if ~isempty(SCR)
            w = B ./ sum(B);
            MCR(klsr_state.t) = sum(w .* SCR);
            MF (klsr_state.t) = sum(w .* (SF.^2)) / sum(w .* SF);
            klsr_state.t = klsr_state.t + 1;
            if klsr_state.t > H, klsr_state.t = 1; end
        end

        %% KLSR 插件
        kopt.fx = f;
        [X, klsr_state, ~] = KLSR(X, f, lb, ub, klsr_state, kopt);

        %%更新 best & curve
        [curBest, idx] = min(f);
        if curBest < bestScore
            bestScore = curBest;
            bestX = X(idx,:);
        end
        curve(G) = bestScore;
    end
end

