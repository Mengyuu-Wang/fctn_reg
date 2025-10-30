function [Y] = cores_2_tensor(cores,sz)

    N = size(cores,1);
    G = pre(cores,N);
    Y = G{1};
    for i = 2:N-1
        Y = tensor_product(Y,G{i});
    end
    G{N} = reshape(G{N},numel(G{N})/sz(N),sz(N));
    Y = Y * G{N};
    Y = reshape(Y,sz);
end

