function [core] = getcore(Z,sz,ranks,n)
    N = numel(sz);
    if n == 1
        szz = horzcat(sz(1),ranks(1,2:end));
        core = reshape(Z,szz);
    elseif n == N
        Z = Z';
        szz = horzcat(ranks(1:N-1,N)',sz(N));
        core = reshape(Z,szz);
    else
        szz = horzcat(sz(n),ranks(1:n-1,n)',ranks(n,n+1:end));
        Z = reshape(Z,szz);
        core = permute(Z,[2:n 1 n+1:N]);
    end

end