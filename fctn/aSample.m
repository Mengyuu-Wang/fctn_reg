function [G_use] = aSample(barG,G,embedding_dims,n)
    N = size(G,1);
    G_use = cell(N,1);
    for i = 1:N
        if i ~= n
            sz = horzcat(embedding_dims(i),size(G{i},2:N));
            G_use{i} = reshape(barG{i},sz);
        end
    end
end