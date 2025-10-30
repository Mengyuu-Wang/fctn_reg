function C = tensor_prod(A,B)
    N = length(size(A));
    C = tensorprod(A,B,2,1);
    dd = horzcat(1:2*N-2);
    for j = 1:2*N-2
        if mod(j,2) == 1
            ddd(j) = dd((j+1)/2);
        else
            ddd(j) = dd(j/2-1+N);
        end
    end
    C = permute(C,ddd);
    ee = horzcat(size(A,1)*size(B,2),size(A,3:N).*size(B,3:N));
    C = reshape(C,ee);
end


%%% exam
%%% A = randn([7 8 9 10]); B = randn([8 6 5 4]);
%%% A = randn([7 8 9]); B = randn([8 6 5]);