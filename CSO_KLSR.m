function [gBestScore, gBest, cg_curve, acc_log] = CSO_KLSR(N, MaxIt, lb, ub, dim, fobj, kopt)
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
    if ~isfield(kopt,'max_evals'),      kopt.max_evals = 2; end  % faithful: the wiped-opts bug forced KLSR's own default of 2
    if ~isfield(kopt,'fit'),            kopt.fit = struct('D',128,'sigma',1.0,'lambda',1e-3, ...
                                           'k',min(3*dim,200),'df',inf,'quality_min',0.6); end
    if ~isfield(kopt,'arch_max'),       kopt.arch_max = 500; end
    if ~isfield(kopt,'warmup'),         kopt.warmup   = max(100, round(0.02*MaxIt)); end
    if ~isfield(kopt,'stallG'),         kopt.stallG   = 20; end
    % --- ablation pass-through (defaults reproduce the deployed operator) ---
    if ~isfield(kopt,'variant'),        kopt.variant  = 'klsr'; end
    if ~isfield(kopt,'qmin'),           kopt.qmin     = 0.6; end
    if ~isfield(kopt,'pscale'),         kopt.pscale   = 0.1; end
    if ~isfield(kopt,'use_pg'),         kopt.use_pg   = true; end
    if ~isfield(kopt,'diag'),           kopt.diag     = false; end
    if ~isfield(kopt,'grad_diag'),      kopt.grad_diag= false; end  % R1.2: log surrogate-vs-true gradient accuracy
    % --- trust-region local sampling for the surrogate fit (redesign) ---
    if ~isfield(kopt,'tr'),             kopt.tr = struct; end
    if ~isfield(kopt.tr,'enable'),      kopt.tr.enable = false; end       % default off = original scattered-archive fit
    if ~isfield(kopt.tr,'M'),           kopt.tr.M      = max(60, 4*dim); end % local design size
    if ~isfield(kopt.tr,'radius'),      kopt.tr.radius = 0.05; end        % half-width as fraction of (ub-lb)
    dbg = struct('fit',0,'trig',0,'call',0,'acc',0,'xev',0);
    global KLSR_GRADLOG; %#ok<GVMIS>  % rows: [t, R2, cos(gTrue,gSurr), |gTrue|]
    
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
    acc_log = zeros(0,2);   % T5: [nfes_at_acceptance, win_src] with win_src 1=mirror, 2=anchor
    
    for i=1:N
        push_archive(X(i,:), F(i));
    end
    
    for t = 1:MaxIt
        if nfes >= max_nfes, cg_curve(t:end) = gBestScore; break; end
        
        xMean = mean(X, 1);
        stopFlag = false;

        % ---- fit mirror ----
        stall_ok = (t - last_gbest_improve) >= kopt.stallG;
        if t >= kopt.warmup && (stall_ok || kopt.grad_diag) ...
           && mod(t,fit_period)==1 && size(arch_X,1) >= max(20,dim+5)
            if kopt.tr.enable && (max_nfes - nfes) > kopt.tr.M
                % dedicated tight local design around the incumbent (accurate gradient)
                M = kopt.tr.M; rad = kopt.tr.radius .* (ub - lb);
                LX = gBest + (rand(M,dim)-0.5)*2 .* rad;
                LX = min(max(LX, lb), ub);
                LF = zeros(M,1);
                for ii = 1:M, LF(ii) = fobj(LX(ii,:)); nfes = nfes + 1; push_archive(LX(ii,:), LF(ii)); end
                fitloc = kopt.fit; fitloc.k = M;
                model = KLSR_fitModel(struct('X',LX,'F',LF), gBest, lb, ub, fitloc);
            else
                archive = struct('X',arch_X,'F',arch_F);
                model = KLSR_fitModel(archive, gBest, lb, ub, kopt.fit);
            end
            if model.has
                dbg.fit = dbg.fit + 1;
                if kopt.grad_diag
                    gT = fd_grad(gBest);                       % true gradient (central finite diff), diagnostic only
                    ngT = max(norm(gT), eps); nS = max(norm(model.n), eps);
                    csv = dot(gT/ngT, model.n(:)'/nS);         % cos(true, surrogate) direction
                    KLSR_GRADLOG(end+1,:) = [t, model.quality, csv, norm(gT)]; %#ok<AGROW>
                end
            end
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
                dbg.trig = dbg.trig + 1;
                remain = max_nfes - nfes;
                if remain > 0
                    dbg.call = dbg.call + 1;
                    opts = struct('fx',Fcand,'alpha',kopt.alpha,...
                                  'max_evals', min(kopt.max_evals, remain), ... % 动态裁剪
                                  'use_pg',kopt.use_pg,'bound',kopt.bound_mode,...
                                  'variant',kopt.variant,'qmin',kopt.qmin,'pscale',kopt.pscale);
                                  
                    [x2,f2,extra_evals,win_src] = KLSR(Xcand, pBest(l,:), gBest, lb, ub, fobj, model, opts);

                    nfes = nfes + extra_evals;
                    dbg.xev = dbg.xev + extra_evals;

                    if f2 < Fcand
                        dbg.acc = dbg.acc + 1;
                        acc_log(end+1,:) = [nfes, win_src]; %#ok<AGROW>  % T5: accepted-injection timing & type
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

    if kopt.diag
        fprintf('[diag] variant=%s fit=%d trig=%d call=%d acc=%d xev=%d\n', ...
                kopt.variant, dbg.fit, dbg.trig, dbg.call, dbg.acc, dbg.xev);
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

    % ===== helper: central finite-difference gradient (diagnostic only, not counted in nfes) =====
    function gT = fd_grad(x0)
        gT = zeros(1, dim);
        h  = 1e-3 * max(ub - lb, 1);          % per-dim step scaled to the search range
        for j = 1:dim
            xp = x0; xm = x0;
            xp(j) = min(x0(j) + h(j), ub(j));
            xm(j) = max(x0(j) - h(j), lb(j));
            gT(j) = (fobj(xp) - fobj(xm)) / max(xp(j) - xm(j), eps);
        end
    end
end