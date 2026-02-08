function [gBestScore, gBest, cg_curve] = PSO_KLSR(N, Max_iteration, lb, ub, dim, fobj, kopt)
    Vmax = 2; noP = N;
    wMax = 0.9; wMin = 0.2; c1 = 2; c2 = 2;
    
    max_nfes = N * Max_iteration;
    nfes = 0;

    iter = Max_iteration;
    vel = zeros(noP, dim);
    pBestScore = inf(noP, 1);
    pBest = zeros(noP, dim);
    gBest = zeros(1, dim);
    cg_curve = zeros(1, iter);
    pos = initialization(noP, dim, ub, lb);
    gBestScore = inf;
    
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

    p_start = kopt.p0; p_end = kopt.p1;
    stall_T = kopt.stall_T; only_on_stall = kopt.only_on_stall;
    alpha_list = kopt.alpha; max_evals_default = kopt.max_evals;
    bound_mode = kopt.bound_mode;
    fit_period = kopt.tau; fit_opts = kopt.fit;
    arch_max = kopt.arch_max; warmup = kopt.warmup; stallG = kopt.stallG;
    model = struct('has',false);
    last_improve = zeros(noP,1);
    last_gbest_improve = 1;

    arch_X = zeros(0,dim); arch_F = zeros(0,1);
    
    for l = 1:iter
        if nfes >= max_nfes, cg_curve(l:end) = gBestScore; break; end
        
        fvals = zeros(noP,1);
        for i = 1:noP
            if nfes >= max_nfes, break; end

            pos(i,:) = max(min(pos(i,:),ub),lb);
            fx = fobj(pos(i,:)); 
            
            nfes = nfes + 1;
            fvals(i) = fx;
            
            if size(arch_X,1) < arch_max
                arch_X(end+1,:) = pos(i,:); arch_F(end+1,1) = fx;
            else
                arch_X = [arch_X(2:end,:); pos(i,:)]; arch_F = [arch_F(2:end); fx];
            end
            if fx < pBestScore(i), pBestScore(i)=fx; pBest(i,:)=pos(i,:); last_improve(i)=l; end
            if fx < gBestScore,    gBestScore=fx;    gBest=pos(i,:);     last_gbest_improve = l; end
        end
        
        if nfes >= max_nfes, cg_curve(l:end) = gBestScore; break; end

        if l >= warmup && (l - last_gbest_improve) >= stallG ...
           && mod(l,fit_period)==1 && size(arch_X,1) >= max(20,dim+5)
            archive = struct('X',arch_X,'F',arch_F);
            model = KLSR_fitModel(archive, gBest, lb, ub, fit_opts);
        end
        
        p_trig = p_start + (l-1)/(iter-1)*(p_end - p_start);
        quota = max(3, ceil(kopt.quota*noP));
        [~,ord] = sort(l - last_improve,'descend');
        idx_try = ord(1:quota);
        
        for i = 1:noP
            if nfes >= max_nfes, break; end
            
            do_it = (rand < p_trig) && ismember(i, idx_try) ...
                    && (~only_on_stall || (l - last_improve(i) >= stall_T));
            
            if do_it
                remain = max_nfes - nfes;
                if remain > 0
                    opts = struct('fx',fvals(i),'alpha',alpha_list,...
                                  'max_evals', min(max_evals_default, remain), ... 
                                  'use_pg',true,'bound',bound_mode);
                                  
                    [x2, f2, extra_evals] = KLSR(pos(i,:), pBest(i,:), gBest, lb, ub, fobj, model, opts);
                    
                    nfes = nfes + extra_evals; 

                    if f2 < fvals(i)
                        pos(i,:) = x2; fvals(i) = f2;
                        if size(arch_X,1) < arch_max
                            arch_X(end+1,:) = x2; arch_F(end+1,1) = f2;
                        else
                            arch_X = [arch_X(2:end,:); x2]; arch_F = [arch_F(2:end); f2];
                        end
                        if f2 < pBestScore(i), pBestScore(i)=f2; pBest(i,:)=x2; last_improve(i)=l; end
                        if f2 < gBestScore,    gBestScore=f2;    gBest=x2;     last_gbest_improve = l; end
                    end
                end
            end
        end
        
        if nfes >= max_nfes, cg_curve(l:end) = gBestScore; break; end

        w = wMax - l * ((wMax - wMin) / iter);
        for i = 1:noP
            r1 = rand(1,dim); r2 = rand(1,dim);
            vel(i,:) = w*vel(i,:) + c1*r1.*(pBest(i,:)-pos(i,:)) + c2*r2.*(gBest-pos(i,:));
            vel(i,:) = max(min(vel(i,:),Vmax), -Vmax);
            pos(i,:) = pos(i,:) + vel(i,:);
        end
        cg_curve(l) = gBestScore;
    end
end