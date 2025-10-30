function X = classical_mode_folding(Xn, sz, k ,N)
%经典mode-n展开的逆运算，不具有一般性

perm_vec = [2:k 1 k+1:N];
perm_vec_sz = [k 1:k-1 k+1:N];
X = permute(reshape(Xn, sz(perm_vec_sz)), perm_vec);

end
