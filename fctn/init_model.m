function [model,cores] = init_model(para, ranks)
% Construct tensor
sz = para.dim;
N = length(sz);
cores = cell(N,1);

for n = 1:N
    if n == 1
        szz = horzcat(sz(n), ranks(n,n+1:end));
        cores{1} = randn(szz);
    elseif n == N
        szz = horzcat(ranks(1:n-1,n)', sz(n));
        cores{N} = randn(szz);
    else
        szz = horzcat(ranks(1:n-1,n)', sz(n), ranks(n,n+1:end));
        cores{n} = randn(szz);
    end    
end
model = cores_2_tensor(cores, sz);

end

