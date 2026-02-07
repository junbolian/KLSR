function model = KLSR_fitModel(archive, center, lb, ub, fit)
% 独立版：核化等值面镜面拟合（RFF + 岭回归）
% 输入:  archive.X(mxd), archive.F(mx1); center(1xd); lb/ub 可标量或1xd
%        fit: struct('D',512,'sigma',1.0,'lambda',1e-3,'k',5d,'df',inf,'quality_min',0.6)
% 输出:  model: struct('has',bool,'c',1xd,'n',1xd,'quality',R2, 'meta',...)

    if nargin < 5, fit = struct; end
    if isempty(archive) || ~isfield(archive,'X') || size(archive.X,1) < 5
        model = struct('has',false,'c',[],'n',[],'quality',0); return;
    end
    if ~isfield(fit,'D'),           fit.D = 512; end
    if ~isfield(fit,'sigma'),       fit.sigma = 1.0; end
    if ~isfield(fit,'lambda'),      fit.lambda = 1e-3; end
    if ~isfield(fit,'k'),           fit.k = min(max(10,5*size(archive.X,2)), size(archive.X,1)); end
    if ~isfield(fit,'df'),          fit.df = inf; end
    if ~isfield(fit,'quality_min'), fit.quality_min = 0.6; end

    X = double(archive.X); F = double(archive.F(:)); d = size(X,2);
    lb = lb(:)'; ub = ub(:)'; rngv = max(ub - lb, eps);

    % 标准化到[0,1]空间
    S  = (X - lb) ./ rngv;
    c0 = (center - lb) ./ rngv;

    % 取近邻样本（可选等值窗口）
    Dists = sqrt(sum((S - c0).^2,2));
    [~,ord] = sort(Dists,'ascend'); pick = ord;
    if isfinite(fit.df)
        mask = abs(F - min(F)) <= fit.df; pick = pick(mask(pick)); if isempty(pick), pick = ord; end
    end
    pick = pick(1:min(fit.k, numel(pick))); S = S(pick,:); y = F(pick);
    k = size(S,1); 
    if k < d+3, model = struct('has',false,'c',[],'n',[],'quality',0); return; end

    % RFF 特征
    D = fit.D;
    Omega = randn(d,D)/fit.sigma;               % d x D
    b     = 2*pi*rand(1,D);                     % 1 x D
    Phi   = sqrt(2/D) * cos(S*Omega + repmat(b,k,1));  % k x D

    % 岭回归闭式解
    A = Phi; w = (A.'*A + fit.lambda*eye(D)) \ (A.'*y);

    % 质量：R^2
    yhat = A*w; ss_res = sum((y - yhat).^2); ss_tot = sum((y - mean(y)).^2) + eps;
    R2 = 1 - ss_res/ss_tot;

    % 在 center 的梯度（映回原坐标）
    sc  = sqrt(2/D)*sin(c0*Omega + b);          % 1 x D
    g_s = - ( (w(:)'.*sc) * Omega.' );          % 1 x d
    g_x = g_s ./ rngv;                          % 链式

    if norm(g_x) < 1e-12 || R2 < fit.quality_min
        model = struct('has',false,'c',[],'n',[],'quality',R2); return;
    end

    n = g_x / norm(g_x);
    model = struct('has',true,'c',center(:)','n',n,'quality',R2, ...
                   'meta',struct('Omega',Omega,'b',b,'w',w,'D',D,'sigma',fit.sigma,'k',k));
end
