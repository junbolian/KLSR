function S3_plot_avg_rank_layout()
% Fig layout:
% Left : Baseline average rank (single blue bar)
% Right: 3D bar over (tau, p0) configs + colorbar
%
% Input : results_sens_tau_p0/raw_ALL.mat
% Output: results_sens_tau_p0/Fig_rank_layout.png, Fig_rank_layout.csv

    inDir = fullfile(pwd, "results_sens_tau_p0");
    load(fullfile(inDir, "raw_ALL.mat"), "raw");

    nF = numel(raw);

    tau_list = raw(1).tau_list(:).';
    p0_list  = raw(1).p0_list(:).';

    nTau = numel(tau_list);
    nP0  = numel(p0_list);

    % ---- per-function medians (baseline + grid) ----
    baseMed = nan(nF,1);
    gridMed = nan(nF, nTau*nP0);

    for fi = 1:nF
        b = raw(fi).baseline.best(:);
        okb = isfinite(b) & ~isnan(b);
        baseMed(fi) = median(b(okb));

        k = 1;
        for it = 1:nTau
            for ip = 1:nP0
                v = squeeze(raw(fi).grid.best(it,ip,:));
                v = v(:);
                ok = isfinite(v) & ~isnan(v);
                gridMed(fi,k) = median(v(ok));
                k = k + 1;
            end
        end
    end

    % ---- rank per function: baseline + all grid ----
    rankBase = nan(nF,1);
    rankGrid = nan(nF, nTau*nP0);

    for fi = 1:nF
        vals = [baseMed(fi), gridMed(fi,:)];
        r = tiedrank(vals);            % rank 1 = best
        rankBase(fi)   = r(1);
        rankGrid(fi,:) = r(2:end);
    end

    avgRankBase = mean(rankBase, "omitnan");      % scalar
    avgRankGrid = mean(rankGrid, 1, "omitnan");   % 1x9

    % reshape avgRankGrid into [nP0 x nTau]
    R = nan(nP0, nTau);
    k = 1;
    for it = 1:nTau
        for ip = 1:nP0
            R(ip,it) = avgRankGrid(k);
            k = k + 1;
        end
    end

    % best config by lowest average rank
    [bestVal, bestIdx] = min(avgRankGrid);
    best_it  = ceil(bestIdx / nP0);
    best_ip  = bestIdx - (best_it-1)*nP0;
    best_tau = tau_list(best_it);
    best_p0  = p0_list(best_ip);

    fprintf("Best avg-rank config: tau=%d, p0=%.2f, avgRank=%.4f\n", best_tau, best_p0, bestVal);

    % ---- style ----
    blueBar = [0.20 0.35 0.62];

    % ---- Plot ----
    fig = figure("Color","w","Position",[160 90 1100 520]);

    % Left: baseline bar
    ax1 = subplot(1,2,1);
    bar(1, avgRankBase, 0.35, 'FaceColor', blueBar, 'EdgeColor', blueBar); % thinner
    grid on; box on;
    xlim([0.5 1.5]);
    set(ax1, "XTick", 1, "XTickLabel", "Baseline");
    ylabel("Average rank");
    title(sprintf("CSO: Baseline (avg over %d functions)", nF), "FontWeight","normal");
    set(ax1, 'Color','w', 'XColor','k', 'YColor','k');

    text(1, avgRankBase, sprintf("%.2f", avgRankBase), ...
        "HorizontalAlignment","center", "VerticalAlignment","bottom", ...
        "FontSize", 10, "FontWeight","bold", "Color","k");

    % Right: 3D bar
    ax2 = subplot(1,2,2);
    bh = bar3(R, 0.95);
    grid on; box on;
    set(ax2, 'Color','w', 'XColor','k', 'YColor','k', 'ZColor','k');

    for kk = 1:numel(bh)
        zdata = bh(kk).ZData;
        bh(kk).FaceColor = 'interp';
        bh(kk).CData     = zdata;
        bh(kk).EdgeColor = [0.35 0.35 0.35];
        bh(kk).LineWidth = 0.8;
    end

    % blue -> white colormap (higher = whiter)
    nC = 256;
    cmap = [linspace(0.10, 1.00, nC)', ...
            linspace(0.35, 1.00, nC)', ...
            linspace(0.85, 1.00, nC)'];
    colormap(ax2, cmap);

    cb = colorbar;
    cb.Label.String = "Average rank (lower is better)";
    cb.Color = 'k';
    cb.Label.Color = 'k';

    xlabel("\tau"); ylabel("p_0"); zlabel("Average rank");
    set(ax2, "XTick", 1:nTau, "XTickLabel", arrayfun(@(x)sprintf("%d",x), tau_list, "uni",0));
    set(ax2, "YTick", 1:nP0,  "YTickLabel", arrayfun(@(x)sprintf("%.2f",x), p0_list,  "uni",0));

    title(sprintf("KLSR-CSO configs (best: \\tau=%d, p_0=%.2f)", best_tau, best_p0), ...
        "FontWeight","normal", "Color","k");

    view([-40 26]);

    % numeric labels on top (black)
    for it = 1:nTau
        for ip = 1:nP0
            z = R(ip,it);
            text(it, ip, z, sprintf("%.2f", z), ...
                "HorizontalAlignment","center", "VerticalAlignment","bottom", ...
                "FontSize", 9, "FontWeight","bold", "Color","k");
        end
    end

    % force all texts black
    set(findall(fig,'Type','text'), 'Color','k');

    outPdf = fullfile(inDir, "Fig_rank_layout.pdf");
    exportgraphics(fig, outPdf, ...
        "ContentType","vector", ...
        "BackgroundColor","white");
    close(fig);


    % ---- Save table ----
    rows = cell(nTau*nP0, 3);
    idx = 1;
    for it = 1:nTau
        for ip = 1:nP0
            rows(idx,:) = {tau_list(it), p0_list(ip), R(ip,it)};
            idx = idx + 1;
        end
    end
    T = cell2table(rows, 'VariableNames', {'tau','p0','AvgRank'});
    outCsv = fullfile(inDir, "Fig_rank_layout.csv");
    writetable(T, outCsv);

    fprintf("Saved: %s\n", outPdf);
    fprintf("Saved: %s\n", outCsv);
end
