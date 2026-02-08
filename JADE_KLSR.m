function [fitnessBestP, bestP, Convergence_curve] = JADE_KLSR(SearchAgents_no, Gmax, lb, ub, dim, fobj, kopt)
% JADE + KLSR

    % Bounds
    if (max(size(ub)) == 1)
        ub = ub .* ones(1, dim);
        lb = lb .* ones(1, dim);
    end
    lu = [lb; ub];
    lb = lb(:)'; ub = ub(:)';

    max_nfes = SearchAgents_no + Gmax * SearchAgents_no;
    nfes = 0;

    % JADE params
    c = 1/10; p = 0.05;
    top = max(1, floor(p * SearchAgents_no));
    A = [];
    uCR = 0.5; uF = 0.5;

    % KLSR params
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
    if ~isfield(kopt,'warmup'),         kopt.warmup   = max(100, round(0.02*Gmax)); end
    if ~isfield(kopt,'stallG'),         kopt.stallG   = 20; end
    
    p_start = kopt.p0;  p_end = kopt.p1;
    fit_period = kopt.tau; fit_opts = kopt.fit;
    alpha_list = kopt.alpha; max_evals_default = kopt.max_evals; % 重命名
    bound_mode = kopt.bound_mode;
    arch_max = kopt.arch_max; warmup = kopt.warmup; stallG = kopt.stallG;
    stall_T = kopt.stall_T; only_on_stall = kopt.only_on_stall;

    % Init
    P = repmat(lb, SearchAgents_no, 1) + rand(SearchAgents_no, dim) .* repmat((ub - lb), SearchAgents_no, 1);
    fitnessP = zeros(1, SearchAgents_no);
    
    for i = 1:SearchAgents_no
        fitnessP(i) = fobj(P(i,:));
        nfes = nfes + 1;
    end
    
    [fitnessBestP, indexBestP] = min(fitnessP);
    bestP = P(indexBestP, :);
    
    % KLSR State
    pBest = P; pBestScore = fitnessP;
    last_improve = zeros(SearchAgents_no, 1);
    last_gbest_improve = 1;
    
    arch_X = P; arch_F = fitnessP(:);
    model = struct('has', false);
    
    Convergence_curve = nan(1, Gmax);
    
    G = 0;
    while G < Gmax
        if nfes >= max_nfes, Convergence_curve(G+1:end) = fitnessBestP; break; end
        G = G + 1;

        % (1) Fit Mirror
        if G >= warmup && (G - last_gbest_improve) >= stallG ...
           && mod(G, fit_period) == 1 && size(arch_X,1) >= max(20, dim+5)
            archive = struct('X', arch_X, 'F', arch_F);
            model = KLSR_fitModel(archive, bestP, lb, ub, fit_opts);
        end
        
        % (2) Trigger
        if Gmax > 1
            p_trig = p_start + (G-1)/(Gmax-1) * (p_end - p_start);
        else
            p_trig = p_start;
        end
        quota = max(3, ceil(kopt.quota * SearchAgents_no));
        [~, ord] = sort(G - last_improve, 'descend');
        idx_try = ord(1:quota);
        
        % (3) CR/F
        CR = uCR + 0.1*randn(1, SearchAgents_no);
        CR(CR>1)=1; CR(CR<0)=0;
        Fv = zeros(1, SearchAgents_no);
        for i=1:SearchAgents_no
            Fi = cauchyrnd(uF, 0.1);
            while Fi <= 0, Fi = cauchyrnd(uF, 0.1); end
            if Fi > 1, Fi = 1; end
            Fv(i) = Fi;
        end
        
        % (4) Top p
        [~, indexSortP] = sort(fitnessP);
        bestTopP = P(indexSortP(1:top), :);
        
        % Loop Agents
        Scr = []; Sf = [];
        
        for i = 1:SearchAgents_no
            if nfes >= max_nfes, break; end

            % Mutation
            k0 = randi(top); Xpbest = bestTopP(k0, :);
            k1 = randi(SearchAgents_no); while k1 == i, k1 = randi(SearchAgents_no); end; P1 = P(k1, :);
            PandA = [P; A];
            k2 = randi(size(PandA,1)); while (k2 == i) || (k2 == k1), k2 = randi(size(PandA,1)); end; P2 = PandA(k2, :);
            
            V = P(i,:) + Fv(i).*(Xpbest - P(i,:)) + Fv(i).*(P1 - P2);
            
            % Crossover
            jrand = randi(dim); U = P(i,:);
            mask = (rand(1,dim) <= CR(i)); mask(jrand) = true;
            U(mask) = V(mask);
            
            % Bound
            for j=1:dim
                while (U(j) > ub(j) || U(j) < lb(j)), U(j) = (ub(j) - lb(j))*rand + lb(j); end
            end
            
            % Evaluation
            fu = fobj(U);
            
            nfes = nfes + 1;
            
            % Archive Push
            arch_X = [arch_X; U]; arch_F = [arch_F; fu];
            if size(arch_X,1) > arch_max
                arch_X = arch_X(end-arch_max+1:end,:);
                arch_F = arch_F(end-arch_max+1:end,:);
            end
            
            % --- KLSR ---
            do_it = (rand < p_trig) && ismember(i, idx_try) ...
                    && (~only_on_stall || (G - last_improve(i) >= stall_T));
            
            if do_it
                remain = max_nfes - nfes;
                if remain > 0
                    opts = struct('fx', fu, 'alpha', alpha_list, ...
                                  'max_evals', min(max_evals_default, remain), ... % 动态
                                  'use_pg', true, 'bound', bound_mode);
                    
                    [u2, fu2, extra_evals] = KLSR(U, pBest(i,:), bestP, lb, ub, fobj, model, opts);
                    
                    nfes = nfes + extra_evals;
                    
                    if fu2 < fu
                        U = u2; fu = fu2;
                        arch_X = [arch_X; U]; arch_F = [arch_F; fu];
                        if size(arch_X,1) > arch_max
                            arch_X = arch_X(end-arch_max+1:end,:);
                            arch_F = arch_F(end-arch_max+1:end,:);
                        end
                    end
                end
            end
            
            % Selection
            if fu < fitnessP(i)
                A = [A; P(i,:)];
                P(i,:) = U; fitnessP(i) = fu;
                Scr(end+1) = CR(i); Sf(end+1) = Fv(i);
                last_improve(i) = G;
                
                if fu < pBestScore(i), pBestScore(i) = fu; pBest(i,:) = P(i,:); end
                if fu < fitnessBestP, fitnessBestP = fu; bestP = P(i,:); last_gbest_improve = G; end
            end
        end
        
        % Archive limit
        if size(A,1) > SearchAgents_no
            rnd_idx = randperm(size(A,1), size(A,1) - SearchAgents_no);
            A(rnd_idx, :) = [];
        end
        
        % Params update
        if ~isempty(Scr)
            newSf = sum(Sf.^2) / (sum(Sf) + eps);
            uCR = (1-c)*uCR + c*mean(Scr);
            uF  = (1-c)*uF  + c*newSf;
        end
        
        Convergence_curve(G) = fitnessBestP;
    end
end

% Helper
function r = cauchyrnd(loc, scale)
    r = loc + scale * tan(pi * (rand - 0.5));
end