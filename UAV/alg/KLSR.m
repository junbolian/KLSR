function [x_new, f_new, extra_evals] = KLSR(x, p, g, lb, ub, fobj, model, opts)
% KLSR  Kernel LevelSet Reflection 
% 先用核化等值面模型的“镜面”做温和反射；若模型质量不足，则退回 p/g 对称反射；全程择优保底。
%
% 输入
%   x      1xd  当前解
%   p      1xd  个人最好（可 []）
%   g      1xd  全局最好（可 []）
%   lb/ub  1xd 或标量
%   fobj   目标函数句柄（最小化）
%   model  结构体：.has(bool) .c(1xd) .n(1xd) .quality(0..1)；若无模型，用 []
%   opts   结构体：.fx([]) .alpha([1 0.5 0.25]) .max_evals(1~2) .use_pg(true) .bound('clip')
%
% 输出
%   x_new, f_new         采用的解及其目标值（择优保底）
%   extra_evals          本次新增评估次数（不含 opts.fx）

    if nargin < 8 || isempty(model), model = struct('has',false); end
    if nargin < 9, opts = struct; end
    if ~isfield(opts,'fx'),        opts.fx = []; end
    if ~isfield(opts,'alpha'),     opts.alpha = [1 0.5 0.25]; end
    if ~isfield(opts,'max_evals'), opts.max_evals = 2; end
    if ~isfield(opts,'use_pg'),    opts.use_pg = true; end
    if ~isfield(opts,'bound'),     opts.bound = 'clip'; end

    x = x(:)';  if isscalar(lb), lb = repmat(lb,size(x)); end
                if isscalar(ub), ub = repmat(ub,size(x)); end

    % baseline
    extra_evals = 0;
    if isempty(opts.fx), fx = fobj(x); extra_evals = extra_evals + 1;
    else, fx = opts.fx; end
    x_best = x; f_best = fx;

    cand = {};

    % 1) 核等值面反射（若模型质量达标）
    if isfield(model,'has') && model.has && model.quality >= 0.6 ...
            && isfield(model,'n') && any(model.n)
        for a = opts.alpha
            y = plane_reflect(x, model.c, model.n, a);
            y = bound_handle(y, lb, ub, opts.bound);
            cand{end+1} = y; %#ok<AGROW>
        end
    end

    % 2) 退路：p/g 对称反射（含一次双反射外推）
    if opts.use_pg
        if ~isempty(p), cand{end+1} = bound_handle(2*p - x, lb, ub, opts.bound); end %#ok<AGROW>
        if ~isempty(g), cand{end+1} = bound_handle(2*g - x, lb, ub, opts.bound); end %#ok<AGROW>
        if ~isempty(p) && ~isempty(g)
            y = 2*p - x; z = 2*g - y;
            cand{end+1} = bound_handle(z, lb, ub, opts.bound); %#ok<AGROW>
        end
    end

    % 3) 预算内评估 & 择优
    budget = opts.max_evals;
    for t = 1:numel(cand)
        if budget <= 0, break; end
        fy = fobj(cand{t});  budget = budget - 1; extra_evals = extra_evals + 1;
        if fy < f_best, f_best = fy; x_best = cand{t}; end
    end

    x_new = x_best; f_new = f_best;
end

% ====== 训练共享镜面：核化等值面（RFF 岭回归），每 τ 代一次 ======
function model = KLSR_fitModel(archive, center, lb, ub, fit)
% archive.X(mxd), archive.F(mx1); center(1xd)
% fit: .D(512) .sigma(1.0) .lambda(1e-3) .k(5d) .df(inf) .quality_min(0.6)
    if nargin < 5, fit = struct; end
    if ~isfield(fit,'D'),           fit.D = 512; end
    if ~isfield(fit,'sigma'),       fit.sigma = 1.0; end
    if ~isfield(fit,'lambda'),      fit.lambda = 1e-3; end
    if ~isfield(fit,'k'),           fit.k = min( max(10, 5*size(archive.X,2)), size(archive.X,1) ); end
    if ~isfield(fit,'df'),          fit.df = inf; end
    if ~isfield(fit,'quality_min'), fit.quality_min = 0.6; end

    model = struct('has',false,'c',[],'n',[],'quality',0);
    if isempty(archive) || ~isfield(archive,'X') || size(archive.X,1) < 5, return; end

    X = double(archive.X); F = double(archive.F(:)); d = size(X,2);
    lb = lb(:)'; ub = ub(:)'; rngv = max(ub - lb, eps);

    % 标准化到 [0,1] 空间，提升稳定性
    S = (X - lb) ./ rngv;  c0 = (center - lb) ./ rngv;

    % 邻域采样
    Dists = sqrt(sum((S - c0).^2,2)); [~,ord] = sort(Dists,'ascend'); pick = ord;
    if isfinite(fit.df)
        mask = abs(F - min(F)) <= fit.df; pick = pick(mask(pick)); if isempty(pick), pick = ord; end
    end
    pick = pick(1:min(fit.k, numel(pick)));  S = S(pick,:); y = F(pick);
    k = size(S,1); if k < d+3, return; end

    % RFF 特征
    D = fit.D;  Omega = randn(d,D)/fit.sigma;  b = 2*pi*rand(1,D);
    Phi = sqrt(2/D)*cos(S*Omega + repmat(b,k,1));

    % 岭回归闭式解
    A = Phi;  w = (A.'*A + fit.lambda*eye(D)) \ (A.'*y);

    % 质量：R^2
    yhat = A*w; ss_res = sum((y - yhat).^2); ss_tot = sum((y - mean(y)).^2) + eps;
    R2 = 1 - ss_res/ss_tot;

    % 在 center 的梯度，映回原坐标
    sc  = sqrt(2/D)*sin(c0*Omega + b);           % 1 x D
    g_s = - ( (w(:)'.*sc) * Omega.' );           % 1 x d
    g_x = g_s ./ rngv;                           % 链式法

    if norm(g_x) < 1e-12 || R2 < fit.quality_min, return; end

    n = g_x / norm(g_x);
    model.has = true; model.c = center(:)'; model.n = n; model.quality = R2;
    model.meta = struct('Omega',Omega,'b',b,'w',w,'D',D,'sigma',fit.sigma,'k',k);
end

% ====== helpers ======
function y = plane_reflect(x, c, n, alpha)
    n = n(:)'; n = n / max(norm(n), eps);
    d = dot( (x - c), n );
    y = x - 2*alpha*d*n;
end

function y = bound_handle(y, lb, ub, how)
    switch lower(how)
        case 'clip'
            y = max(min(y, ub), lb);
        case 'mirror'
            range = ub - lb; t = mod(y - lb, 2*range); y = lb + abs(t - range);
        otherwise
            y = max(min(y, ub), lb);
    end
end
