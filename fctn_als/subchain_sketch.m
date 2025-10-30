function X = subchain_sketch(G,n)
    N = size(G,1);
    if n == 1
        X = G{2};
        for j = 3:N
            X = half_tensor_product(X,G{j});
        end    
    else
        X = G{1};
        for j = 2:n-1
            X = half_tensor_product(X,G{j});
        end
        for j = n+1:N
            X = half_tensor_product(X,G{j});
        end
    end

end