function C = x_prod(A,B,num)    
    N = length(size(A));
    C = tensorprod(A,B,[1 num],[1 num]);
    dd = 1:2*N-4;
    for j = 1:2*N-4
        if mod(j,2) == 1
            ddd(j) = dd((j+1)/2);
        else
            ddd(j) = dd(j/2+N-2);
        end
    end
    C = permute(C,ddd);
    ee = horzcat(size(A,2:N/2).*size(B,2:N/2),size(A,N/2+2:N).*size(B,N/2+2:N));
    C = reshape(C,ee);
end


%%% exam
%%% A = randn([4 5 6 7]); B = randn([4 2 6 3]);
%%% A = randn([4 5 6 7 8 9]); B = randn([4 2 3 7 5 6]);