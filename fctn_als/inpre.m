function[trcores] = inpre(G,cores)
%% 特异性  对第N次变形
    N = size(G,1); 
    trcores = cell(N,1);
    
    trcores{1} = permute(G{1},[N 1:N-1]);
    if N == 3
        trcores{2} = permute(G{2},[3 1 2]);
        trcores{3} = permute(G{3},[3 2 1]);
    elseif N == 4
        trcores{2} = permute(G{2},[4 1 2 3]);
        trcores{3} = permute(G{3},[4 2 1 3]);
        trcores{4} = permute(G{4},[3 2 1]);
        trcores{4} = reshape(trcores{4},size(cores{4}));
    end
end