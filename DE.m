function [BestScore, BestPos, BestCost] = DE(nPop, MaxIt, VarMin, VarMax, nVar, CostFunction)
%% Problem Definition
VarSize = [1 nVar]; % Decision Variables Matrix Size

%% DE Parameters
F = 0.5;        % Fixed Scaling Factor
pCR = 0.9;      % Crossover Probability

%% Initialization
empty_individual.Position = [];
empty_individual.Cost = [];

BestSol.Cost = inf;

pop = repmat(empty_individual, nPop, 1);
for i = 1:nPop
    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
    pop(i).Cost = CostFunction(pop(i).Position);
    
    if pop(i).Cost < BestSol.Cost
        BestSol = pop(i);
    end
end

BestCost = zeros(MaxIt, 1);

%% DE Main Loop
for it = 1:MaxIt
    
    for i = 1:nPop
        
        x = pop(i).Position;
        
        % Select 3 different individuals (excluding i)
        A = randperm(nPop);
        A(A==i) = [];
        
        a = A(1);
        b = A(2);
        c = A(3);
        
        % Mutation (DE/rand/1)
        y = pop(a).Position + F * (pop(b).Position - pop(c).Position);
        
        % Boundary constraint handling
        y = max(y, VarMin);
        y = min(y, VarMax);
        
        % Crossover
        z = zeros(size(x));
        j0 = randi([1 numel(x)]);
        
        for j = 1:numel(x)
            if j == j0 || rand <= pCR
                z(j) = y(j);
            else
                z(j) = x(j);
            end
        end
        
        % Selection
        NewSol.Position = z;
        NewSol.Cost = CostFunction(NewSol.Position);
        
        if NewSol.Cost < pop(i).Cost
            pop(i) = NewSol;
            
            if pop(i).Cost < BestSol.Cost
                BestSol = pop(i);
            end
        end
        
    end
    
    % Update Best Cost
    BestCost(it) = BestSol.Cost;
    
end

% Final Results
BestScore = BestSol.Cost;
BestPos = BestSol.Position;

end