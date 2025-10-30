function [X] = gengengen(sz, ranks, noise, large_elem)
% Construct tensor
N = length(sz);
cores = cell(N,1);
for n = 1:N
    if n == 1
        szz = horzcat(sz(n), ranks(n,n+1:end));
        cores{1} = randn(szz);
        if large_elem > 0
            r0 = randsample(max(ranks(n,n+1:end)),N-n);
            szz = randsample(sz(n),1);
            szzz = horzcat(szz, r0');
            cores{1}(szzz) = large_elem;
        end
    elseif n == N
        szz = horzcat(ranks(1:n-1,n)', sz(n));
        cores{N} = randn(szz);
        if large_elem > 0
            r0 = randsample(max(ranks(1:n-1,n)),N-1);
            szz = randsample(sz(n),1);
            szzz = horzcat(r0',szz);
            cores{N}(szzz) = large_elem;
        end
    else
        szz = horzcat(ranks(1:n-1,n)', sz(n), ranks(n,n+1:end));
        cores{n} = randn(szz);
        if large_elem > 0
            r1 = randsample(max(ranks(1:n-1,n)),n-1);
            r2 = randsample(max(ranks(n,n+1:end)),N-n);
            szz = randsample(sz(n),1);
            szzz = horzcat(r1,szz,r2);
            cores{N}(szzz) = large_elem;
        end
    end    
end

X = cores_2_tensor(cores, sz);
X = X + noise*randn(sz);

end
