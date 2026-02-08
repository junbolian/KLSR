function [gBestScore, gBest, cg_curve] = CSO(N, MaxIt, lb, ub, dim, fobj)

    if isscalar(lb), lb = lb * ones(1, dim); end
    if isscalar(ub), ub = ub * ones(1, dim); end

    % --- CSO typical setting ---
    phi  = 0.15;         % learning rate to mean position
    Vmax = 2.0;          % velocity clip

    max_nfes = N + floor(N/2) * MaxIt;
    nfes = 0;

    % --- init ---
    X = repmat(lb, N, 1) + rand(N, dim) .* repmat(ub - lb, N, 1);
    V = zeros(N, dim);
    F = zeros(N,1);
    
    for i=1:N
        F(i) = fobj(X(i,:));
        nfes = nfes + 1;
    end
    
    [gBestScore, idx] = min(F);
    gBest = X(idx,:);
    cg_curve = zeros(1, MaxIt);

    for t = 1:MaxIt
        if nfes >= max_nfes, cg_curve(t:end) = gBestScore; break; end
        
        % --- random pairing ---
        perm = randperm(N);
        half = floor(N/2);
        A = perm(1:half);
        B = perm(half+1:2*half);
        
        xMean = mean(X, 1);
        
        % --- competition update ---
        for k = 1:half
            if nfes >= max_nfes, break; end
            
            i1 = A(k); i2 = B(k);
            
            % decide winner/loser
            if F(i1) <= F(i2)
                w = i1; l = i2;
            else
                w = i2; l = i1;
            end
            
            r1 = rand(1,dim); r2 = rand(1,dim); r3 = rand(1,dim);
            
            % loser velocity & position update
            V(l,:) = r1.*V(l,:) + r2.*(X(w,:) - X(l,:)) + phi*r3.*(xMean - X(l,:));
            V(l,:) = max(min(V(l,:), Vmax), -Vmax);
            X(l,:) = X(l,:) + V(l,:);
            
            % bound clip
            X(l,:) = max(min(X(l,:), ub), lb);
            
            % evaluate loser only
            F(l) = fobj(X(l,:));
            
            nfes = nfes + 1;
        end
        
        % update global best
        [bestNow, idx] = min(F);
        if bestNow < gBestScore
            gBestScore = bestNow;
            gBest = X(idx,:);
        end
        cg_curve(t) = gBestScore;
    end
end