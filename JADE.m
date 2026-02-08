function [fitnessBestP, bestP, Convergence_curve] = JADE(SearchAgents_no, Gmax, lb, ub, dim, fobj)
% JADE: Adaptive Differential Evolution with Optional External Archive

    % Bounds
    if (max(size(ub)) == 1)
        ub = ub .* ones(1, dim);
        lb = lb .* ones(1, dim);
    end
    lu = [lb; ub];

    max_nfes = SearchAgents_no + Gmax * SearchAgents_no;
    nfes = 0;

    % Parameters
    c = 1/10;
    p = 0.05;
    top = max(1, floor(p * SearchAgents_no));
    
    A = []; 
    
    uCR = 0.5;
    uF  = 0.5;

    % Init
    P = repmat(lu(1, :), SearchAgents_no, 1) + rand(SearchAgents_no, dim) .* (repmat(lu(2, :) - lu(1, :), SearchAgents_no, 1));
    fitnessP = zeros(1, SearchAgents_no);
    
    for i=1:SearchAgents_no
        fitnessP(i) = fobj(P(i,:));
        nfes = nfes + 1;
    end

    % Main Loop
    G = 0;
    Convergence_curve = nan(1, Gmax);
    
    % Find best
    [fitnessBestP, indexBestP] = min(fitnessP);
    bestP = P(indexBestP, :);

    while G < Gmax
        if nfes >= max_nfes, Convergence_curve(G+1:end) = fitnessBestP; break; end
        G = G + 1;

        Scr = []; Sf = [];
        
        % Generate CR / F
        CR = uCR + 0.1*randn(1, SearchAgents_no);
        CR(CR > 1) = 1; CR(CR < 0) = 0; % Simple clip for speed, or re-sample
        
        F = zeros(1, SearchAgents_no);
        for i=1:SearchAgents_no
            Fi = cauchyrnd(uF, 0.1);
            while Fi <= 0, Fi = cauchyrnd(uF, 0.1); end
            if Fi > 1, Fi = 1; end
            F(i) = Fi;
        end

        % Sort for p-best
        [~, indexSortP] = sort(fitnessP);
        bestTopP = P(indexSortP(1:top), :);

        % Mutation + Crossover + Selection
        for i=1:SearchAgents_no
            if nfes >= max_nfes, break; end

            % Mutation
            k0 = randi(top);
            Xpbest = bestTopP(k0, :);
            
            k1 = randi(SearchAgents_no);
            while k1 == i, k1 = randi(SearchAgents_no); end
            P1 = P(k1, :);
            
            PandA = [P; A];
            k2 = randi(size(PandA,1));
            while (k2 == i) || (k2 == k1), k2 = randi(size(PandA,1)); end
            P2 = PandA(k2, :);
            
            V = P(i,:) + F(i).*(Xpbest - P(i,:)) + F(i).*(P1 - P2);
            
            % Crossover
            jrand = randi(dim);
            U = P(i,:);
            mask = (rand(1,dim) <= CR(i));
            mask(jrand) = true;
            U(mask) = V(mask);
            
            % Bound Check
            for j=1:dim
                while (U(j) > ub(j) || U(j) < lb(j))
                    U(j) = (ub(j) - lb(j))*rand + lb(j);
                end
            end
            
            % Evaluation
            fitnessU = fobj(U);
            
            nfes = nfes + 1;
            
            % Selection
            if fitnessU < fitnessP(i)
                A = [A; P(i,:)];
                P(i,:) = U;
                fitnessP(i) = fitnessU;
                Scr(end+1) = CR(i); %#ok<AGROW>
                Sf(end+1)  = F(i);  %#ok<AGROW>
                
                if fitnessU < fitnessBestP
                    fitnessBestP = fitnessU;
                    bestP = U;
                end
            end
        end
        
        % Archive limit
        if size(A,1) > SearchAgents_no
            rnd_idx = randperm(size(A,1), size(A,1) - SearchAgents_no);
            A(rnd_idx, :) = [];
        end
        
        % Update uCR / uF
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