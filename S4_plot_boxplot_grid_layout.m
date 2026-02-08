function S4_plot_boxplot_grid_layout()
% Single-page PDF with 24 boxplots

    inDir = fullfile(pwd, "results_sens_tau_p0");
    load(fullfile(inDir, "raw_ALL.mat"), "raw");

    nF = numel(raw);

    %% -------- compute ImprovePct and best grid per function --------
    func_id = nan(nF,1);
    improve = nan(nF,1);
    best_it = nan(nF,1);
    best_ip = nan(nF,1);

    for fi = 1:nF
        F = raw(fi).func_id;
        func_id(fi) = F;

        tau_list = raw(fi).tau_list;
        p0_list  = raw(fi).p0_list;

        base = raw(fi).baseline.best(:);
        base = base(isfinite(base) & ~isnan(base));
        base_med = median(base);

        bm = Inf; bit = 1; bip = 1;
        for it = 1:numel(tau_list)
            for ip = 1:numel(p0_list)
                v = squeeze(raw(fi).grid.best(it,ip,:));
                v = v(:);
                ok = isfinite(v) & ~isnan(v);
                if ~any(ok), continue; end
                mv = median(v(ok));
                if mv < bm
                    bm = mv; bit = it; bip = ip;
                end
            end
        end

        best_it(fi) = bit;
        best_ip(fi) = bip;

        if isfinite(base_med) && base_med ~= 0
            improve(fi) = (base_med - bm) / base_med * 100;
        else
            improve(fi) = NaN;
        end
    end

    %% -------- choose 24 functions --------
    absImp = abs(improve);
    absImp(~isfinite(absImp)) = -Inf;
    [~, ord] = sort(absImp, 'descend');
    keep_idx = ord(1:24);

    [~, kord] = sort(func_id(keep_idx));
    keep_idx = keep_idx(kord);

    %% -------- layout parameters --------
    nCols = 4;
    nRows = 6;

    fontName = "Helvetica";
    baseFS   = 8;             % ticks
    titleFS  = 9;             % subplot title
    ylabFS   = 8;             % ylabel
    supFS    = 12;            % top title
    capFS    = 12;            % bottom caption

    % Canvas similar to paper figure proportions
    fig = figure('Color','w','Units','pixels','Position',[50 50 1100 1250]);

    t = tiledlayout(nRows, nCols, ...
        'TileSpacing','compact', ...
        'Padding','compact');

    % Global defaults
    set(fig, 'DefaultAxesFontName', fontName);
    set(fig, 'DefaultTextFontName', fontName);
    set(fig, 'DefaultAxesFontSize', baseFS);
    set(fig, 'DefaultTextFontSize', baseFS);
    set(fig, 'DefaultAxesLineWidth', 0.8);
    set(fig, 'DefaultAxesTickDir', 'out');
    set(fig, 'DefaultAxesXColor', 'k');
    set(fig, 'DefaultAxesYColor', 'k');
    set(fig, 'DefaultTextColor',  'k');


    %% -------- plotting --------
    epsv = 1e-300;

    for k = 1:numel(keep_idx)
        fi = keep_idx(k);
        F  = raw(fi).func_id;

        base = raw(fi).baseline.best(:);
        base = base(isfinite(base) & ~isnan(base));

        bit = best_it(fi);
        bip = best_ip(fi);

        kbest = squeeze(raw(fi).grid.best(bit,bip,:));
        kbest = kbest(:);
        kbest = kbest(isfinite(kbest) & ~isnan(kbest));

        base_plot  = max(base,  epsv);
        kbest_plot = max(kbest, epsv);

        ax = nexttile;

        % Outliers as red "+" boxes blue; medians red; whiskers gray dashed
        boxplot([base_plot, kbest_plot], ...
            'Labels', {'Baseline','KLSR(best)'}, ...
            'Symbol','r+', ...
            'Whisker', 1.5, ...
            'Widths', 0.55);

        set(ax,'YScale','log');
        ax.Box   = 'on';
        ax.Layer = 'top';

        ax.Color = 'w';
        
        ax.XGrid = 'on';
        ax.YGrid = 'on';
        ax.XMinorGrid = 'on';
        ax.YMinorGrid = 'on';
        
        ax.GridColor      = [0.70 0.70 0.70];
        ax.MinorGridColor = [0.85 0.85 0.85];
        
        ax.GridLineStyle      = '-';
        ax.MinorGridLineStyle = ':';
        
        ax.GridAlpha      = 0.5;
        ax.MinorGridAlpha = 0.3;

        title(ax, sprintf('F%02d (Dim=50)', F), ...
            'FontWeight','normal', 'FontSize', titleFS);

        ylabel(ax, 'Best Fitness (log)', 'FontSize', ylabFS);

        % Tight but stable log limits
        allv = [base_plot(:); kbest_plot(:)];
        allv = allv(isfinite(allv) & allv > 0);
        if ~isempty(allv)
            lo = 10^(floor(log10(min(allv))) - 0.1);
            hi = 10^(ceil (log10(max(allv))) + 0.1);
            % ylim(ax, [lo, hi]);
        end

        % Make x labels slightly smaller to avoid crowding
        ax.XAxis.FontSize = baseFS-1;

        % ---- Post-style tweaks ----
        % Blue boxes
        hBox = findobj(ax,'Tag','Box');
        set(hBox,'Color',[0 0 1],'LineWidth',0.8);

        % Red medians
        hMed = findobj(ax,'Tag','Median');
        set(hMed,'Color',[1 0 0],'LineWidth',0.8);

        % Gray dashed whiskers + adjacent values
        hWhisk = findobj(ax,'Tag','Whisker');
        set(hWhisk,'LineStyle','--','Color',[0.5 0.5 0.5],'LineWidth',0.8);

        hAdj = findobj(ax,'Tag','Adjacent Value');
        set(hAdj,'LineStyle','--','Color',[0.5 0.5 0.5],'LineWidth',0.8);

        % Ensure outliers are red "+"
        hOut = findobj(ax,'Tag','Outliers');
        set(hOut,'Marker','+','MarkerEdgeColor',[1 0 0],'LineWidth',0.8);
    end

    %% -------- export --------
    outPdf = fullfile(inDir, "Fig_boxplot_grid_24.pdf");
    exportgraphics(fig, outPdf, 'ContentType','vector');
    close(fig);

    fprintf("Saved: %s\n", outPdf);
end
