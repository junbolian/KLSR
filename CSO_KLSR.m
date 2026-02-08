function [gBestScore, gBest, cg_curve] = CSO_KLSR(N, MaxIt, lb, ub, dim, fobj, kopt)
% CSO + KLSR

    if isscalar(lb), lb = lb * ones(1, dim); end
    if isscalar(ub), ub = ub * ones(1, dim); end

    % ===== CSO params =====
    phi  = 0.15;
    Vmax = 2.0;

    % ===== KLSR defaults =====
    if nargin < 7 || isempty(kopt), kopt = struct; end
    if ~isfield(kopt,'tau'),            kopt.tau = 50; end
    if ~isfield(kopt,'p0'),             kopt.p0  = 0.30; end
    if ~isfield(kopt,'p1'),             kopt.p1  = 0.10; end
    if ~isfield(kopt,'stall_T'),        kopt.stall_T = 7; end
    if ~isfield(kopt,'only_on_stall'),  kopt.only_on_stall = true; end
    if ~isfield(kopt,'quota'),          kopt.quota = 0.10; end
    if ~isfield(kopt,'bound_mode'),     kopt.bound_mode = 'clip'; end
    if ~isfield(kopt,'alpha'),          kopt.alpha = [1 0.5 0.25]; end
    if ~isfield(kopt,'max_evals'),      kopt.max_evals = 1; end
    if ~isfield(kopt,'fit'),            kopt.fit = struct('D',128,'sigma',1.0,'lambda',1e-3, ...
                                           'k',min(3*dim,200),'df',inf,'quality_min',0.6); end
    if ~isfield(kopt,'arch_max'),       kopt.arch_max = 500; end
    if ~isfield(kopt,'warmup'),         kopt.warmup   = max(100, round(0.02*MaxIt)); end
    if ~isfield(kopt,'stallG'),         kopt.stallG   = 20; end
    
    p_start = kopt.p0; p_end = kopt.p1;
    fit_period = kopt.tau;
    stall_T = kopt.stall_T;

    max_nfes = N + floor(N/2) * MaxIt;
    nfes = 0;

    % ===== init swarm =====
    X = repmat(lb, N, 1) + rand(N, dim) .* repmat(ub - lb, N, 1);
    V = zeros(N, dim);
    F = zeros(N,1);
    
    for i=1:N
        F(i) = fobj(X(i,:));
        nfes = nfes + 1;
    end
    
    % personal best (for KLSR fallback)
    pBest = X;
    pBestScore = F;
    
    [gBestScore, idx] = min(F);
    gBest = X(idx,:);
    cg_curve = zeros(1, MaxIt);

    % ===== KLSR state =====
    last_improve = ones(N,1);
    last_gbest_improve = 1;
    arch_X = zeros(0,dim); arch_F = zeros(0,1);
    model = struct('has',false);
    
    for i=1:N
        push_archive(X(i,:), F(i));
    end
    
    for t = 1:MaxIt
        if nfes >= max_nfes, cg_curve(t:end) = gBestScore; break; end
        
        xMean = mean(X, 1);
        stopFlag = false;

        % ---- fit mirror ----
        if t >= kopt.warmup && (t - last_gbest_improve) >= kopt.stallG ...
           && mod(t,fit_period)==1 && size(arch_X,1) >= max(20,dim+5)
            archive = struct('X',arch_X,'F',arch_F);
            model = KLSR_fitModel(archive, gBest, lb, ub, kopt.fit);
        end
        
        % ---- trigger schedule ----
        p_trig = p_start + (t-1)/(MaxIt-1)*(p_end - p_start);
        quota = max(3, ceil(kopt.quota * N));
        [~,ord] = sort(t - last_improve,'descend');
        idx_try = ord(1:quota);
        
        % ---- CSO pairing ----
        perm = randperm(N);
        half = floor(N/2);
        A = perm(1:half);
        B = perm(half+1:2*half);
        
        for k = 1:half
            if nfes >= max_nfes, stopFlag = true; break; end
            
            i1 = A(k); i2 = B(k);
            
            % winner/loser
            if F(i1) <= F(i2), w = i1; l = i2;
            else,              w = i2; l = i1;
            end
            
            r1 = rand(1,dim); r2 = rand(1,dim); r3 = rand(1,dim);
            
            % loser update
            V(l,:) = r1.*V(l,:) + r2.*(X(w,:) - X(l,:)) + phi*r3.*(xMean - X(l,:));
            V(l,:) = max(min(V(l,:), Vmax), -Vmax);
            Xcand  = X(l,:) + V(l,:);
            Xcand  = max(min(Xcand, ub), lb);
            
            Fcand = fobj(Xcand);
            
            nfes = nfes + 1;
            push_archive(Xcand, Fcand);
            
            do_klsr = (rand < p_trig) && ismember(l, idx_try) ...
                      && (~kopt.only_on_stall || (t - last_improve(l) >= stall_T));
            
            if do_klsr
                remain = max_nfes - nfes;
                if remain > 0
                    opts = struct('fx',Fcand,'alpha',kopt.alpha,...
                                  'max_evals', min(kopt.max_evals, remain), ... % 动态裁剪
                                  'use_pg',true,'bound',kopt.bound_mode);
                                  
                    [x2,f2,extra_evals] = KLSR(Xcand, pBest(l,:), gBest, lb, ub, fobj, model, opts);
                    
                    nfes = nfes + extra_evals;

                    if f2 < Fcand
                        Xcand = x2; Fcand = f2;
                        push_archive(Xcand, Fcand);
                    end
                end
            end
            
            % accept loser move
            X(l,:) = Xcand;
            F(l)   = Fcand;
            
            % update pBest
            if F(l) < pBestScore(l)
                pBestScore(l) = F(l);
                pBest(l,:)    = X(l,:);
                last_improve(l)= t;
            end
            
            % update gBest
            if F(l) < gBestScore
                gBestScore = F(l);
                gBest = X(l,:);
                last_gbest_improve = t;
            end
        end
        
        if stopFlag, cg_curve(t:end) = gBestScore; break; end
        
        cg_curve(t) = gBestScore;
    end
    
    % ===== helper: sliding archive =====
    function push_archive(xrow, fval)
        if size(arch_X,1) < kopt.arch_max
            arch_X(end+1,:) = xrow;
            arch_F(end+1,1) = fval;
        else
            arch_X = [arch_X(2:end,:); xrow];
            arch_F = [arch_F(2:end); fval];
        end
    end
end