function C = tensor_product(A,B)
    N = length(size(A));
    M = length(size(B));
    SS = max(M,N);
    C = tensorprod(A,B,2,2);
    if M >= N
        dd = horzcat(1:2*N-2);
        for j = 1:2*N-2
            if mod(j,2) == 1
                ddd(j) = dd((j+1)/2);
            else
                ddd(j) = dd(j/2-1+N);
            end
        end
        ddd = horzcat(ddd,length(ddd)+1:M+N-2);
    else
        dd = horzcat(1:2*M-2);
        for j = 1:length(dd)
            if mod(j,2) == 1
                ddd(j) = dd((j+1)/2);
            else
                ddd(j) = dd(j/2-1+M)+N-M;
            end
        end
        ddd = horzcat(ddd,M:N-1);
    end
    C = permute(C,ddd);
    ee = horzcat(size(A,1)*size(B,1),size(A,3:SS).*size(B,3:SS));
    C = reshape(C,ee);
end


%%% exam
%%% A = randn([7 8 9 10]); B = randn([6 8 5 4]);
%%% A = randn([7 8 9 10]); B = randn([6 8 2]);
%%% A = randn([7 8 3]); B = randn([6 8 5 4]);