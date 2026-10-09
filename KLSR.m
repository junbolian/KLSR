function [x_new, f_new, extra_evals, win_src] = KLSR(x, p, g, lb, ub, fobj, model, opts)
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

    if nargin < 7 || isempty(model), model = struct('has',false); end
    if nargin < 8 || isempty(opts),  opts  = struct; end
    if ~isfield(opts,'fx'),        opts.fx = []; end
    if ~isfield(opts,'alpha'),     opts.alpha = [1 0.5 0.25]; end
    if ~isfield(opts,'max_evals'), opts.max_evals = 2; end
    if ~isfield(opts,'use_pg'),    opts.use_pg = true; end
    if ~isfield(opts,'bound'),     opts.bound = 'clip'; end
    % --- ablation controls (default = deployed behavior; backward compatible) ---
    if ~isfield(opts,'variant'),   opts.variant = 'klsr'; end   % klsr|klsr_pure|rand_reflect|rand_perturb|anchor|grad_step
    if ~isfield(opts,'qmin'),      opts.qmin = 0.6;      end     % surrogate quality gate (R^2)
    if ~isfield(opts,'pscale'),    opts.pscale = 0.1;    end     % rand_perturb step as fraction of (ub-lb)

    x = x(:)';  if isscalar(lb), lb = repmat(lb,size(x)); end
                if isscalar(ub), ub = repmat(ub,size(x)); end

    % baseline
    extra_evals = 0;
    if isempty(opts.fx), fx = fobj(x); extra_evals = extra_evals + 1;
    else, fx = opts.fx; end
    x_best = x; f_best = fx;

    cand = {};
    n_mirror = 0;   % number of mirror (surrogate-reflection) candidates, for acceptance-type logging
    d = numel(x);
    model_ok = isfield(model,'has') && model.has && model.quality >= opts.qmin ...
               && isfield(model,'n') && any(model.n);

    switch lower(opts.variant)
        case 'klsr'          % deployed operator: surrogate level-set reflection + p/g fallback
            if model_ok
                for a = opts.alpha
                    cand{end+1} = bound_handle(plane_reflect(x, model.c, model.n, a), lb, ub, opts.bound); %#ok<AGROW>
                end
            end
            n_mirror = numel(cand);   % candidates so far are mirror-based
            if opts.use_pg
                cand = add_anchor(cand, x, p, g, lb, ub, opts.bound);
            end

        case 'klsr_pure'     % surrogate level-set reflection ONLY (isolates the geometry from the fallback)
            if model_ok
                for a = opts.alpha
                    cand{end+1} = bound_handle(plane_reflect(x, model.c, model.n, a), lb, ub, opts.bound); %#ok<AGROW>
                end
            end

        case 'rand_reflect'  % reflection across a hyperplane with a RANDOM normal (isolates the learned orientation)
            center = x; if ~isempty(g), center = g; elseif model_ok, center = model.c; end
            for a = opts.alpha
                nr = randn(1,d); nr = nr / max(norm(nr), eps);
                cand{end+1} = bound_handle(plane_reflect(x, center, nr, a), lb, ub, opts.bound); %#ok<AGROW>
            end

        case 'rand_perturb'  % isotropic Gaussian step, matched scale, no hyperplane (isolates "any extra move")
            step = opts.pscale * (ub - lb);
            for a = opts.alpha
                cand{end+1} = bound_handle(x + a*step.*randn(1,d), lb, ub, opts.bound); %#ok<AGROW>
            end

        case 'anchor'        % symmetric reflection about personal/global best only (no surrogate)
            cand = add_anchor(cand, x, p, g, lb, ub, opts.bound);

        case 'grad_step'     % descent step ALONG the surrogate gradient, matched length (no reflection)
            if model_ok
                n  = model.n(:)'; n = n / max(norm(n), eps);
                d0 = dot(x - model.c, n);
                sgn = grad_descent_sign(x, n, model, lb, ub);
                L  = 2*abs(d0);                          % match the reflection displacement magnitude
                for a = opts.alpha
                    cand{end+1} = bound_handle(x + sgn*a*L*n, lb, ub, opts.bound); %#ok<AGROW>
                end
            end

        case 'gbest'         % global-best symmetric reflection ONLY (the effective anchor move)
            if ~isempty(g)
                cand{end+1} = bound_handle(2*g - x, lb, ub, opts.bound); %#ok<AGROW>
            end

        case 'gbest_geo'     % SALVAGE: keep 2g-x first, then add surrogate reflection (geometry no longer crowds out 2g-x)
            if ~isempty(g)
                cand{end+1} = bound_handle(2*g - x, lb, ub, opts.bound); %#ok<AGROW>
            end
            if model_ok
                for a = opts.alpha
                    cand{end+1} = bound_handle(plane_reflect(x, model.c, model.n, a), lb, ub, opts.bound); %#ok<AGROW>
                end
            end

        otherwise
            error('KLSR:unknownVariant', 'Unknown opts.variant: %s', opts.variant);
    end

    % 3) 预算内评估 & 择优
    budget = opts.max_evals; win_idx = 0;
    for t = 1:numel(cand)
        if budget <= 0, break; end
        fy = fobj(cand{t});  budget = budget - 1; extra_evals = extra_evals + 1;
        if fy < f_best, f_best = fy; x_best = cand{t}; win_idx = t; end
    end

    x_new = x_best; f_new = f_best;
    if win_idx == 0,            win_src = 0;
    elseif win_idx <= n_mirror, win_src = 1;   % mirror (surrogate-reflection) candidate won
    else,                       win_src = 2;   % anchor (opposition) candidate won
    end
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

function cand = add_anchor(cand, x, p, g, lb, ub, how)
% p/g symmetric reflection (+ one double-reflection extrapolation), original order
    if ~isempty(p), cand{end+1} = bound_handle(2*p - x, lb, ub, how); end %#ok<AGROW>
    if ~isempty(g), cand{end+1} = bound_handle(2*g - x, lb, ub, how); end %#ok<AGROW>
    if ~isempty(p) && ~isempty(g)
        y = 2*p - x; z = 2*g - y;
        cand{end+1} = bound_handle(z, lb, ub, how); %#ok<AGROW>
    end
end

function s = grad_descent_sign(x, n, model, lb, ub)
% Pick the sign along n that decreases the fitted surrogate (descent direction).
    s = -1;  % fallback: n is the (up-)gradient direction, so descent is -n
    if isfield(model,'meta') && ~isempty(model.meta)
        m = model.meta; rngv = max(ub - lb, eps); h = 1e-3;
        sp = ((x + h*n) - lb)./rngv;  sm = ((x - h*n) - lb)./rngv;
        vp = (sqrt(2/m.D)*cos(sp*m.Omega + m.b)) * m.w;
        vm = (sqrt(2/m.D)*cos(sm*m.Omega + m.b)) * m.w;
        if vp <= vm, s = +1; else, s = -1; end
    end
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
