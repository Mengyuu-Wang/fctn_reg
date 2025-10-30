function X = subchain_matrix(cores,n)
    N = size(cores,1);
    G = pre(cores,n);
    if n == 1
        X = G{2};
        for j = 3:N
            X = tensor_product(X,G{j});
        end    
    else
        X = G{1};
        for j = 2:n-1
            X = tensor_product(X,G{j});
        end
        for j = n+1:N
            X = tensor_product(X,G{j});
        end
    end

end