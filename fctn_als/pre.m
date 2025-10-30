function[G] = pre(cores,n)
%%%%验证4阶成功
    N = size(cores,1);
    G = cell(N,1);
    for i = 1:N
        G{i} = permute(cores{i},[1:n-1 n+1:N n]);
    end
    for i = 1:n-1
        if i == 1
        else
            szz = horzcat(prod(size(G{i},1:i-1)),size(G{i},i:N));
            G{i} = reshape(G{i},szz);
            sz = length(size(G{i}));
            G{i} = permute(G{i},[2 1 3:sz]);   
        end
    end
    for i = n+1:N
        if i == 2
        else
            szz = horzcat(prod(size(G{i},1:i-2)),size(G{i},i-1:N));
            G{i} = reshape(G{i},szz);
            sz = length(size(G{i}));
            G{i} = permute(G{i},[2 1 3:sz]); 
        end
    end
end