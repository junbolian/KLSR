% Particle Swarm Optimization
function [gBestScore, gBest, cg_curve] = PSO(N, Max_iteration, lb, ub, dim, fobj)
    % PSO Information
    Vmax = 2;
    noP = N;
    wMax = 0.9;
    wMin = 0.2;
    c1 = 2;
    c2 = 2;
    
    max_nfes = N * Max_iteration;
    nfes = 0;

    % Initializations
    iter = Max_iteration;
    vel = zeros(noP, dim);
    pBestScore = inf(noP, 1);
    pBest = zeros(noP, dim);
    gBest = zeros(1, dim);
    cg_curve = zeros(1, iter);
    
    % Initialize search history and fitness history
    search_history = zeros(noP, iter, dim);  
    fitness_history = zeros(noP, iter);  
    % Random initialization for agents
    pos = initialization(noP, dim, ub, lb); 
    % Initialize gBestScore for a minimization problem
    gBestScore = inf;
    
    for l = 1:iter 
        if nfes >= max_nfes
            cg_curve(l:end) = gBestScore;
            break;
        end
        
        for i = 1:noP
            if nfes >= max_nfes, break; end

            % Return back the particles that go beyond the boundaries
            Flag4ub = pos(i,:) > ub;
            Flag4lb = pos(i,:) < lb;
            pos(i,:) = (pos(i,:) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            
            % Calculate objective function
            fitness = fobj(pos(i,:));
            
            nfes = nfes + 1;
            
            fitness_history(i, l) = fitness;  
            
            % Update personal best
            if pBestScore(i) > fitness
                pBestScore(i) = fitness;
                pBest(i, :) = pos(i, :);
            end
            
            % Update global best
            if gBestScore > fitness
                gBestScore = fitness;
                gBest = pos(i, :);
            end
            
            % Store the current position
            search_history(i, l, :) = pos(i, :);
        end
        
        % Update the W of PSO
        w = wMax - l * ((wMax - wMin) / iter);
        
        % Update the Velocity and Position of particles
        for i = 1:noP
            for j = 1:dim       
                vel(i,j) = w * vel(i,j) + c1 * rand() * (pBest(i,j) - pos(i,j)) + c2 * rand() * (gBest(j) - pos(i,j));
                
                % Limit the velocity
                if vel(i,j) > Vmax, vel(i,j) = Vmax; end
                if vel(i,j) < -Vmax, vel(i,j) = -Vmax; end            
                pos(i,j) = pos(i,j) + vel(i,j);
            end
        end
        cg_curve(l) = gBestScore;
    end
end