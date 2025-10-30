function [barG] = bSample(G,n)
    N = size(G,1);
    barG = cell(N,1);
    for i = 1:N
        if i ~= n && i~= N
            szz = horzcat(size(G{i},1),prod(size(G{i},2:N-1)),size(G{i},N));
            barG{i} = reshape(G{i},szz);
        elseif i == N
            barG{i} = G{i};
        end
    end
end