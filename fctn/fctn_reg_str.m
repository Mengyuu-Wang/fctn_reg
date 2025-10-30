function [best_model,best_runtime,trainerror ] = fctn_reg_str( para, rank, X,Y, Xv, Yv )
%% set parameters
iterTotal = para.maxiter;
shakyRate = 1.5;
N=para.N;
L=para.L;
M=para.M;
lambda=para.lambda;
NN = size(X,1);

%% initialization
%initialize the random model
Replicates = 5;  
best_valerror = inf;

for r = 1:Replicates
    % 随机初始化
    [tmp_model,tmp_cores] = init_model(para,rank);
    re_cores = cell(N,1);
    pp = cell(N,1);

    % 初始化误差
    model = tmp_model;
    cores = tmp_cores;
    minTrainingFuncValue = calcobjfunction(para, model, X, Y);
    minValidationFuncValue = calcobjfunction(para, model, Xv, Yv);
    iterStart = 0;
    trainerror = zeros(para.maxiter,1);  % 注意这里放在 for 循环内重新分配
    
    tic;

%% main loop
for iter = iterStart+1:iterTotal
    
    %% update first L dimensions  
    if L == 1
        for i = 2:N
            re_cores{i} = permute(cores{i},[2:N 1]);
        end
        for i = 4:N
            szz = horzcat(prod(size(re_cores{i},1:N-2)),size(re_cores{i},N-1:N));
            re_cores{i} = reshape(re_cores{i},szz);
        end
        %% Structural
        pp{2} = tensorprod(re_cores{2},re_cores{2},1,1);
        for i = 3:N
            pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
        end

        xx = classical_mode_unfolding(X,2) * classical_mode_unfolding(X,2)';
        vv = pp{2};
        for i = 3:N
            vv = x_prod(vv,pp{i},N-i+3);
        end
        V = re_cores{2};
        for i = 3:N
            V = tensor_prod(V,re_cores{i});
        end

        AA = kron(xx,vv);
        HH = lambda * kron(vv,eye(para.P(1)));
        
        yy = classical_mode_unfolding(Y,1);
        AT = tensorprod(yy,V,2,1);
        AT = tensorprod(X,AT,1,1);
        AT = reshape(AT,numel(AT),1);
        
        zz = (AA + HH) \ (AT);
        cores{1} = reshape(zz,size(cores{1}));
    else
        for j = 1:L
            if j == 1
                % pre_process
                for i = 2:L
                    re_cores{i} = permute(cores{i},[2:N 1]);
                end
                for i = 4:L
                    szz = horzcat(prod(size(re_cores{i},1:i-2)),size(re_cores{i},i-2:N));
                    re_cores{i} = reshape(re_cores{i},szz);
                end
                for i = L+1:N
                    szz = horzcat(size(cores{i},1),prod(size(cores{i},2:L)),size(cores{i},L+1:N));
                    re_cores{i} = reshape(cores{i},szz);
                    re_cores{i} = permute(re_cores{i},[3:M+2 1:2]);
                end
                for i = L+3:N
                    szz = horzcat(prod(size(re_cores{i},1:i-1-L)),size(re_cores{i},i-L:M+2));
                    re_cores{i} = reshape(re_cores{i},szz);
                end
                % compute u>  v
                U_right = re_cores{2};
                for i = 3:L
                    U_right = tensor_prod(U_right,re_cores{i});
                end
                szz = horzcat(size(U_right,1),prod(size(U_right,2:M+1)),size(U_right,M+2));
                U_right = reshape(U_right,szz);
                V = re_cores{L+1};
                for i = L+2:N
                    V = tensor_prod(V,re_cores{i});
                end
                xx = reshape(X,NN,size(X,2),prod(size(X,3:L+1)));
                C = tensorprod(xx,U_right,3,1);

                %% Stru
                pp{2} = tensorprod(re_cores{2},re_cores{2},1,1);
                for i = 3:L
                    pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
                end
                pp{L+1} = tensorprod(re_cores{L+1},re_cores{L+1},1,1);
                for i = L+2:N
                    pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
                end
            
                uu = pp{2};
                for i = 3:L
                    uu = x_prod(uu,pp{i},N-i+3);
                end
                szz = horzcat(prod(size(uu,1:ndims(uu)/2-1)),size(uu,ndims(uu)/2), ...
                    prod(size(uu,1:ndims(uu)/2-1)),size(uu,ndims(uu)/2));
                uu = reshape(uu,szz);

                vv = pp{L+1};
                for i = L+2:N
                    vv = x_prod(vv,pp{i},M-i+L+4);
                end
                

                % compute coef matrix
                HH = tensorprod(uu,vv,[1 3],[2 4]);
                HH = permute(HH,[1 3 2 4]);
                szz = size(HH,1) * size(HH,2);
                HH = reshape(HH,szz,szz);
                HH = lambda * kron(HH,eye(para.P(1)));
                cc = tensorprod(C,C,1,1);
                AA = tensorprod(cc,vv,[2 5],[2 4]);
                AA = permute(AA,[1 2 5 3 4 6]);
                szz = szz * para.P(1);
                AA = reshape(AA,szz,szz);
                
                yy = classical_mode_unfolding(Y,1);
                AT = tensorprod(yy,C,1,1);
                AT = tensorprod(AT,V,[1 3],[1 3]);
                AT = reshape(AT,numel(AT),1);
                
                zz = (AA + HH) \ (AT);
                cores{1} = reshape(zz,size(cores{1}));
            elseif j == L
                % pre_process
                for i = 1:L-1
                    re_cores{i} = cores{i};
                end
                for i = 3:L-1
                    szz = horzcat(prod(size(cores{i},1:i-1)),size(cores{i},i:N));
                    re_cores{i} = reshape(re_cores{i},szz);
                end
                for i = L+1:N
                    szz = horzcat(prod(size(cores{i},1:L-1)),size(cores{i},L),size(cores{i},L+1:N));
                    re_cores{i} = reshape(cores{i},szz);
                    re_cores{i} = permute(re_cores{i},[3:N-L+2 1:2]);
                end
                for i = L+3:N
                    szz = horzcat(prod(size(re_cores{i},1:i-1-L)),size(re_cores{i},i-L:M+2));
                    re_cores{i} = reshape(re_cores{i},szz);
                end
                % compute u<  v
                U_left = re_cores{1};
                for i = 2:L-1
                    U_left = tensor_prod(U_left,re_cores{i});
                end
                szz = horzcat(size(U_left,1),size(U_left,2),prod(size(U_left,3:M+2)));
                U_left = reshape(U_left,szz);
                V = re_cores{L+1};
                for i = L+2:N
                    V = tensor_prod(V,re_cores{i});
                end
                xx = reshape(X,NN,prod(size(X,2:L)),size(X,L+1));
                C = tensorprod(xx,U_left,2,1);

                %% Stru
                pp{1} = tensorprod(re_cores{1},re_cores{1},1,1);
                for i = 2:L-1
                    pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
                end
                pp{L+1} = tensorprod(re_cores{L+1},re_cores{L+1},1,1);
                for i = L+2:N
                    pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
                end
            
                uu = pp{1};
                for i = 2:L-1
                    uu = x_prod(uu,pp{i},N-i+2);
                end
                szz = horzcat(size(uu,1),prod(size(uu,2:ndims(uu)/2)), ...
                    size(uu,1),prod(size(uu,2:ndims(uu)/2)));
                uu = reshape(uu,szz);
                vv = pp{L+1};
                for i = L+2:N
                    vv = x_prod(vv,pp{i},M-i+L+4);
                end
                
                HH = tensorprod(uu,vv,[2 4],[1 3]);
                HH = permute(HH,[1 3 2 4]);
                szz = size(HH,1) * size(HH,2);
                HH = reshape(HH,szz,szz);
                HH = lambda * kron(HH,eye(para.P(L)));
                cc = tensorprod(C,C,1,1);
                AA = tensorprod(cc,vv,[3 6],[1 3]);
                AA = permute(AA,[1 2 5 3 4 6]);
                szz = szz * para.P(L);
                AA = reshape(AA,szz,szz);
                
                yy = classical_mode_unfolding(Y,1);
                AT = tensorprod(yy,C,1,1);
                AT = tensorprod(AT,V,[1 4],[1 2]);
                AT = reshape(AT,numel(AT),1);
                
                zz = (AA + HH) \ (AT);
                szz = horzcat(size(cores{L},j),prod(size(cores{L},1:j-1)),prod(size(cores{L},j+1:N)));
                zz = reshape(zz,szz);
                zz = permute(zz,[2 1 3]);
                cores{L} = reshape(zz,size(cores{L}));

            else
                % pre_process
                for i = 1:j-1
                    re_cores{i} = cores{i};
                end
                for i = 3:j-1
                    szz = horzcat(prod(size(cores{i},1:i-1)),size(cores{i},i:N));
                    re_cores{i} = reshape(re_cores{i},szz);
                end
                for i = j+1:L
                    szz = horzcat(prod(size(cores{i},1:j-1)),size(cores{i},j),size(cores{i},j+1:N));
                    re_cores{i} = reshape(cores{i},szz);
                    re_cores{i} = permute(re_cores{i},[3:length(re_cores) 1:2]);
                end
                for i = L+1:N
                    szz = horzcat(prod(size(cores{i},1:j-1)),size(cores{i},j),prod(size(cores{i},j+1:L)),size(cores{i},L+1:N));
                    re_cores{i} = reshape(cores{i},szz);
                    re_cores{i} = permute(re_cores{i},[4:3+M 1:3]);
                end
                for i = L+3:N
                    szz = horzcat(prod(size(re_cores{i},1:i-1-L)),size(re_cores{i},i-L:M+3));
                    re_cores{i} = reshape(re_cores{i},szz);
                end
                % compute u<  u>  v
                U_left = re_cores{1};
                for i = 2:j-1
                    U_left = tensor_prod(U_left,re_cores{i});
                end
                szz = horzcat(size(U_left,1),size(U_left,2),prod(size(U_left,3:L-j+2)),prod(size(U_left,L-j+3:N-j+2)));
                U_left = reshape(U_left,szz);
                U_right = re_cores{j+1};
                for i = j+2:L
                    U_right = tensor_prod(U_right,re_cores{i});
                end
                szz = horzcat(size(U_right,1),prod(size(U_right,2:M+1)),size(U_right,M+2),size(U_right,M+3));
                U_right = reshape(U_right,szz);
                V = re_cores{L+1};
                for i = L+2:N
                    V = tensor_prod(V,re_cores{i});
                end
                szz = horzcat(NN,size(X,2:j),size(X,j+1),size(X,j+2:L+1));
                xx = reshape(X,szz);
                C = tensorprod(xx,U_left,2,1);
                C = tensorprod(C,U_right,[3 5],[1 3]);
                
                %% Stru
                pp{1} = tensorprod(re_cores{1},re_cores{1},1,1);
                for i = 2:j-1
                    pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
                end
                pp{j+1} = tensorprod(re_cores{j+1},re_cores{j+1},1,1);
                for i = j+2:L
                    pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
                end
                pp{L+1} = tensorprod(re_cores{L+1},re_cores{L+1},1,1);
                for i = L+2:N
                    pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
                end
            
                uu_left = pp{1};
                for i = 2:j-1
                    uu_left = x_prod(uu_left,pp{i},N-i+2);
                end
                szz = horzcat(size(uu_left,1),prod(size(uu_left,2:L-j+1)),prod(size(uu_left,L-j+2:ndims(uu_left)/2)), ...
                    size(uu_left,1),prod(size(uu_left,2:L-j+1)),prod(size(uu_left,L-j+2:ndims(uu_left)/2)));
                uu_left = reshape(uu_left,szz);

                uu_right = pp{j+1};
                for i = j+2:L
                    uu_right = x_prod(uu_right,pp{i},N-i+j+2);
                end
                szz = horzcat(prod(size(uu_right,1:ndims(uu_right)/2-2)),size(uu_right,ndims(uu_right)/2-1),size(uu_right,ndims(uu_right)/2), ...
                    prod(size(uu_right,1:ndims(uu_right)/2-2)),size(uu_right,ndims(uu_right)/2-1),size(uu_right,ndims(uu_right)/2));
                uu_right = reshape(uu_right,szz);

                vv = pp{L+1};
                for i = L+2:N
                    vv = x_prod(vv,pp{i},M-i+L+5);
                end
                
                HH = tensorprod(uu_left,uu_right,[2 5],[2 5]);
                HH = tensorprod(HH,vv,[2 4 5 7],[1 4 3 6]);
                HH = permute(HH,[1 3 5 2 4 6]);
                szz = size(HH,1) * size(HH,2) *size(HH,3);
                HH = reshape(HH,szz,szz);
                HH = lambda * kron(HH,eye(para.P(j)));
                cc = tensorprod(C,C,1,1);
                AA = tensorprod(cc,vv,[3 4 8 9],[1 3 4 6]);
                AA = permute(AA,[1 2 3 7 4 5 6 8]);
                szz = szz * para.P(j);
                AA = reshape(AA,szz,szz);
                
                yy = classical_mode_unfolding(Y,1);
                AT = tensorprod(yy,C,1,1);
                AT = tensorprod(AT,V,[1 4 5],[1 2 4]);
                AT = reshape(AT,numel(AT),1);
                
                zz = (AA + HH) \ (AT);
                szz = horzcat(size(cores{j},j),prod(size(cores{j},1:j-1)),prod(size(cores{j},j+1:N)));
                zz = permute(reshape(zz,szz),[2 1 3]);
                cores{j} = reshape(zz,size(cores{j}));  
            end 
        end
     end
    
    
    %% update last M dimensions  
    if M == 1
        % pre_process
        for i = 1:2
            re_cores{i} = cores{i};
        end
        for i = 3:N-1
            szz = horzcat(prod(size(cores{i},1:i-1)),size(cores{i},i:N)) ;
            re_cores{i} = reshape(cores{i},szz);
        end
        % compute U
        U = re_cores{1};
        for i = 2:N-1
            U = tensor_prod(U,re_cores{i});
        end

        %% Structural
        pp{1} = tensorprod(re_cores{1},re_cores{1},1,1);
        for i = 2:N-1
            pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
        end
        
        uu = pp{1};
        for i = 2:N-1
            uu = x_prod(uu,pp{i},N-i+2);
        end
        
        BB = uu;
        FF = lambda * uu;
        x = classical_mode_unfolding(X,1);
        y = classical_mode_unfolding(Y,1);
        YB = x * U;
        YB = tensorprod(y,YB,1,1);  

        zz = YB / (BB + FF);
        cores{N} = reshape(zz,size(cores{N}));
    else
        for i = 1:2
            re_cores{i} = cores{i};
        end
        for i = 3:L
            szz = horzcat(prod(size(cores{i},1:i-1)),size(cores{i},i:N)) ;
            re_cores{i} = reshape(cores{i},szz);
        end
        % compute U_old
        UU = re_cores{1};
        for i = 2:L
            UU = tensor_prod(UU,re_cores{i});
        end
        
        pp{1} = tensorprod(re_cores{1},re_cores{1},1,1);
        for i = 2:L
            pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
        end
        
        uu_old = pp{1};
        for i = 2:L
            uu_old = x_prod(uu_old,pp{i},N-i+2);
        end
        

        for k = 1:M
            if k == 1
                %pro_process
                for i = L+2:N
                    szz = horzcat(prod(size(cores{i},1:L)),size(cores{i},L+1:N));
                    re_cores{i} = reshape(cores{i},szz);
                    re_cores{i} = permute(re_cores{i},[3:M+1 1 2]);
                end
                % compute v> and update U
                V_right = re_cores{L+2};
                for i = L+3:N
                    V_right = tensor_prod(V_right,re_cores{i});
                end
                szz = horzcat(size(UU,1),size(UU,2),prod(size(UU,3:M+1)));
                U = reshape(UU,szz); 
                x = reshape(X,NN,numel(X)/NN);
                D = tensorprod(x,U,2,1);

                %% Stru
                pp{L+2} = tensorprod(re_cores{L+2},re_cores{L+2},1,1);
                for i = L+3:N
                    pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
                end
                
                vv = pp{L+2};
                for i = L+3:N
                    vv = x_prod(vv,pp{i},N-i+L+3);
                end
                szz = horzcat(size(uu_old,1),prod(size(uu_old,2:M)));
                sz = horzcat(szz,szz);
                uu = reshape(uu_old,sz);

                % compute coef matrix
                FF = tensorprod(uu,vv,[2 4],[1 3]);
                FF = permute(FF,[1 3 2 4]);
                szz = size(FF,1) * size(FF,2);
                FF = reshape(FF,szz,szz);
                FF = lambda * FF;

                DD = tensorprod(D,D,1,1);
                BB = tensorprod(DD,vv,[2 4],[1 3]);
                BB = permute(BB,[1 3 2 4]);
                BB = reshape(BB,szz,szz);

                szz = horzcat(NN,size(Y,2),prod(size(Y,3:M+1)));
                y = reshape(Y,szz);
                YB = tensorprod(y,D,1,1);
                YB = tensorprod(YB,V_right,[2 4],[1 2]);
                YB = reshape(YB,size(YB,1),numel(YB)/size(YB,1));
        
                zz = YB / (BB + FF);
                szz = horzcat(size(cores{k+L},k+L),prod(size(cores{k+L},1:k+L-1)),prod(size(cores{k+L},k+L+1:N)));
                zz = permute(reshape(zz,szz),[2 1 3]);
                cores{k+L} = reshape(zz,size(cores{k+L}));
                
            elseif k == M
                %pro_process
                for i = L+1:N-1
                    szz = horzcat(prod(size(cores{i},1:L)),size(cores{i},L+1:N));
                    re_cores{i} = reshape(cores{i},szz);
                    re_cores{i} = permute(re_cores{i},[2:M+1 1]);
                end
                % compute v< and update U
                V_left = re_cores{L+1};
                for i = L+2:N-1
                    V_left = tensor_prod(V_left,re_cores{i});
                end
                szz = horzcat(size(UU,1),prod(size(UU,2:M)),size(UU,M+1));
                U = reshape(UU,szz);
                x = reshape(X,NN,numel(X)/NN);
                D = tensorprod(x,U,2,1);
                
                %% Stru
                pp{L+1} = tensorprod(re_cores{L+1},re_cores{L+1},1,1);
                for i = L+2:N-1
                    pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
                end
                szz = horzcat(prod(size(uu_old,1:M-1)),size(uu_old,M));
                sz = horzcat(szz,szz);
                uu = reshape(uu_old,sz);

                vv = pp{L+1};
                for i = L+2:N-1
                    vv = x_prod(vv,pp{i},M-i+L+3);
                end

                % compute coef matrix
                FF = tensorprod(uu,vv,[1 3],[2 4]);
                FF = permute(FF,[1 3 2 4]);
                szz = size(FF,1) * size(FF,2);
                FF = reshape(FF,szz,szz);
                FF = lambda * FF;

                DD = tensorprod(D,D,1,1);
                BB = tensorprod(DD,vv,[1 3],[2 4]);
                BB = permute(BB,[1 3 2 4]);
                BB = reshape(BB,szz,szz);

                szz = horzcat(NN,prod(size(Y,2:M)),size(Y,M+1));
                y = reshape(Y,szz);
                YB = tensorprod(y,D,1,1);
                YB = tensorprod(YB,V_left,[1 3],[1 3]);
                YB = reshape(YB,size(YB,1),numel(YB)/size(YB,1));
        
                zz = YB / (BB + FF);
                cores{k+L} = reshape(zz',size(cores{k+L}));
            else
                %pro_process
                for i = L+1:L+k-1
                    szz = horzcat(prod(size(cores{i},1:L)),size(cores{i},L+1:N));
                    re_cores{i} = reshape(cores{i},szz);
                    re_cores{i} = permute(re_cores{i},[2:M+1 1]);
                end
                for i = L+k+1:N
                    szz = horzcat(prod(size(cores{i},1:L)),prod(size(cores{i},L+1:L+k-1)),size(cores{i},L+k:N));
                    re_cores{i} = reshape(cores{i},szz);
                    re_cores{i} = permute(re_cores{i},[4:M+1 1:3]);
                end
                % compute v< v> and update U
                V_left = re_cores{L+1};
                for i = L+2:L+k-1
                    V_left = tensor_prod(V_left,re_cores{i});
                end
                V_right = re_cores{L+k+1};
                for i = L+k+2:N
                    V_right = tensor_prod(V_right,re_cores{i});
                end
                szz = horzcat(size(UU,1),prod(size(UU,2:k)),size(UU,k+1),prod(size(UU,k+2:M+1)));
                U = reshape(UU,szz);
                x = reshape(X,NN,numel(X)/NN);
                D = tensorprod(x,U,2,1);
                
                %% Stru
                pp{L+1} = tensorprod(re_cores{L+1},re_cores{L+1},1,1);
                for i = L+2:L+k-1
                    pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
                end
                pp{L+k+1} = tensorprod(re_cores{L+k+1},re_cores{L+k+1},1,1);
                for i = L+k+2:N
                    pp{i} = tensorprod(re_cores{i},re_cores{i},2,2);
                end
                szz = horzcat(prod(size(uu_old,1:k-1)), ...
                    size(uu_old,k),prod(size(uu_old,k+1:M)));
                sz = horzcat(szz,szz);
                uu = reshape(uu_old,sz);
                
                vv_left = pp{L+1};
                for i = L+2:L+k-1
                    vv_left = x_prod(vv_left,pp{i},k-i+L);
                end
                vv_right = pp{L+k+1};
                for i = L+k+2:N
                    vv_right = x_prod(vv_right,pp{i},M-k-i+L+7);
                end
                
                % compute coef matrix
                FF = tensorprod(uu,vv_left,[1 4],[3 6]);
                FF = tensorprod(FF,vv_right,[2 4 6 8],[1 4 2 5]);
                FF = permute(FF,[1 3 5 2 4 6]);
                szz = size(FF,1) * size(FF,2) * size(FF,3);
                FF = reshape(FF,szz,szz);
                FF = lambda * FF;

                DD = tensorprod(D,D,1,1);
                BB = tensorprod(DD,vv_left,[1 4],[3 6]);
                BB = tensorprod(BB,vv_right,[2 4 6 8],[1 4 2 5]);
                BB = permute(BB,[1 3 5 2 4 6]);
                BB = reshape(BB,szz,szz);

                szz = horzcat(NN,prod(size(Y,2:k)),size(Y,k+1),prod(size(Y,k+2:M+1)));
                y = reshape(Y,szz);
                YB = tensorprod(y,D,1,1);
                YB = tensorprod(YB,V_left,[1 4],[1 4]);
                YB = tensorprod(YB,V_right,[2 4 6],[1 2 3]);
                YB = reshape(YB,size(YB,1),numel(YB)/size(YB,1));
        
                zz = YB / (BB + FF);
                szz = horzcat(size(cores{k+L},k+L),prod(size(cores{k+L},1:k+L-1)),prod(size(cores{k+L},k+L+1:N)));
                zz = permute(reshape(zz,szz),[2 1 3]);
                cores{k+L} = reshape(zz,size(cores{k+L}));
            end
        end
    end
    
    
    %% compute error
    model = cores_2_tensor(cores,para.dim);    
   
    trainingFuncValue = calcobjfunction(para,model,X ,Y);
    trainerror(iter)=trainingFuncValue;
    if abs(trainingFuncValue - minTrainingFuncValue)/minTrainingFuncValue<=1e-3
        break;
    end      
    if trainingFuncValue <= shakyRate * minTrainingFuncValue
        minTrainingFuncValue = min(minTrainingFuncValue, trainingFuncValue);
        disp('descening');
    else
        disp('not descening, error');
        break;
    end
       
    validationFuncValue = calcobjfunction(para,model,Xv,Yv);
    
    disp(['    Iter: ', num2str(iter), '. Training: ', num2str(trainingFuncValue), '. Validation: ', num2str(validationFuncValue)]);
    
    minValidationFuncValue = min(minValidationFuncValue, validationFuncValue);
   
    
end
runtime = toc;
disp('Train the model. Finish.');

    % 用验证误差判断是否更新最优模型
    est_testy = contract(Xv, model, para.L);
    Ypred = zscore(est_testy(:));
    Ytrue = zscore(Yv(:));
    error =  sqrt(mean((Ytrue - Ypred).^2));      
    if error < best_valerror
        best_model = model;
        %best_cores = cores;
        best_runtime = runtime;
        best_valerror = error;
    end
end