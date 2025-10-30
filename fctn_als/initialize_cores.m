function cores = initialize_cores(sz, ranks)
%initialize_cores Initializes cores using Gaussian distribution
% cores = initialize_cores(sz, ranks); returns a length-N cell with N cores,
%each with entires drawn iid from the standard Gaussian distribution. sz is
%a length-N vector with the sizes, and ranks is a matrix with the
%outgoing ranks.

% Main code
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

end
