function C = half_tensor_product(A,B)
    N = length(size(A));
    sz1 = size(A,1);
    sz2 = size(A,3:N);
    sz3 = size(B,3:N);
    szz = horzcat(sz1,size(A,2),prod(sz2));
    A = permute(reshape(A,szz),[3 2 1]);
    szz = horzcat(size(B,1),size(B,2),prod(sz3));
    B = permute(reshape(B,szz),[2 3 1]);
    C = mtimesx(A,B);
    C = permute(C,[3 1 2]);
    szz = horzcat(sz1,sz2,sz3);
    C = reshape(C,szz);
    %%
    dd = horzcat(1:2*N-4);
    for j = 1:2*N-4
        if mod(j,2) == 1
            ddd(j) = dd((j+1)/2);
        else
            ddd(j) = dd(j/2-2+N);
        end
    end
    ddd = horzcat(0,ddd)+1;
    C = permute(C,ddd);
    szz = sz2 .* sz3;
    szz = horzcat(sz1,szz);
    C = reshape(C,szz);
end

%%% exam
%%% A = randn([6 8 9]); B = randn([6 8 5]);
%%% A = randn([6 8 9 10]); B = randn([6 8 5 4]);
%%% addpath("mtimesx\mtimesx_20110223\");

