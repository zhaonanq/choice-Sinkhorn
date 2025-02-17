clear; clc; close all;

rng(2024);
mlist = 50:50:500;
nlist = mlist * 2;

nsize = length(mlist);

ntest = 12;
bnds = zeros(nsize, 1);

for i = 1:nsize
    
    m = mlist(i);
    n = nlist(i);
    bndnow = zeros(ntest, 1);
    
    parfor k = 1:ntest
        A = abs(sprandn(m, n, 0.8));
        p = rand(m, 1);
        q = rand(n, 1);
        q = q / sum(q) * sum(p);
        bound = getskcond(A, p, q);
        if bound > 0
            bndnow(k) = bound;
        end % End if
    end % End for
    
    bnds(i) = median(nonzeros(bndnow));
    
end % End for

plot(mlist, bnds, 'LineWidth', 3, 'Marker', '+', 'MarkerSize', 3);
xlabel("Size of m");
ylabel("Theoretical bound");
set(gca, 'FontSize', 20, 'LineWidth', 1, 'Box', 'on');

% saveas(gca, "bound-sublin.pdf");

