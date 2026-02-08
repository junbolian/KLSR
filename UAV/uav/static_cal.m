count = sum(static_index);
median_rep_index = zeros(1,500);
best_rep_index = zeros(1,500);
worst_rep_index = zeros(1,500);
for i  = 1 : 500
    median_rep_index(i) = median(find(static_index(:, i) == 1));
    best_rep_index(i) = min(find(static_index(:, i) == 1));
    worst_rep_index(i) = max(find(static_index(:, i) == 1));
end
