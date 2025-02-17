function [bnd1] = getskcond(A, p, q)

[~, n] = size(A);
v = rand(n, 1);

for i = 1:500
    u = p ./ (A * v);
    v = q ./ (A' * u);
end % End for

uopt = u;
vopt = v;
UAV = diag(u) * A * diag(v);

if max(uopt) > 1e+01 || max(vopt) >= 1e+01
    bnd1 = -1;
    return;
end % End if

% Construct Laplacian
A = UAV;
Ae = sum(A, 2);
ATe = sum(A, 1);

L = [diag(Ae), -A;
    -A', diag(ATe)];

eval = eig(full(L));

l = min(max(p), max(q));
ub1 = max(uopt) / min(uopt);
ub2 = max(uopt) * max(vopt);
ub3 = 1 / (min([uopt; vopt]));
bnd1 = max([ub1, ub2, ub3]) * l / eval(2);

end % End function