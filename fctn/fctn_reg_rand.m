function [best_model,best_runtime,trainerror ] = fctn_reg_rand(para, rank, J, X,Y, Xv, Yv )
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
    [tmp_model,tmp_cores] = init_model(para,rank);
    re_cores = cell(N,1);
    G = cell(N,1);
    sampling_probs = cell(M,1);
    GG = cell(N,1);
    embedding_dims = J * ones(NN,1);

    % 初始化误差
    model = tmp_model;
    cores = tmp_cores;
    minTrainingFuncValue = calcobjfunction(para,model, X,Y);
    minValidationFuncValue = calcobjfunction(para,model,Xv,Yv);
    iterStart = 0;
    trainerror = zeros(para.maxiter,1);

sz = para.Q;
slow_idx = cell(1,M);
sz_shifted = [1 sz(1:end-1)];
idx_prod = cumprod(sz_shifted);
sz_pts = cell(1,M);
for n = 1:M
    sz_pts{n} = round(linspace(0, sz(n), 2));
    slow_idx{n} = cell(1,1);
    J = embedding_dims(n);
    samples_lin_idx_2 = prod(sz_shifted(1:n))*(sz_pts{n}(1):sz_pts{n}(2)-1).';
    slow_idx{n}{1} = repelem(samples_lin_idx_2, J, 1);
end

disp('Train the model. Running...');
tic;

%% main loop
for iter = iterStart+1:iterTotal
    
    %% update first L dimensions  
    if L == 1
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

                G{1} = re_cores{L+1};
                for i = L+2:N
                    G{i-L} = permute(re_cores{i},[2 1 3:length(re_cores{i})]);
                end
                for i = 1:M
                    U = col(classical_mode_unfolding(G{i}, 1));
                    sampling_probs{i} = sum(abs(U).^2, 2)/size(U, 2);
                end

                samples = nan(J, M);
                barG = bSample(G,N);
                for m = 1:M
                    samples(:, m) = randsample(sz(m), J, true, sampling_probs{m});
                    GG{m} = barG{m}(samples(:,m), :, :);
                end
                core_samples = aSample(GG,G,embedding_dims,N);

                rescaling = ones(J,1);
                for m = 1:M
                    rescaling = rescaling ./ sqrt(sampling_probs{m}(samples(:, m)));
                end
                rescaling = rescaling ./ sqrt(J);

                VV = core_samples{1};
                for i = 2:M
                    VV = half_tensor_product(VV,core_samples{i});
                end

                szzz = size(VV);
                VV = classical_mode_unfolding(VV,1);
                VV = rescaling .* VV;
                VV = reshape(VV,szzz);
                
                yy = zeros(NN,J);
                for i = 1:NN
                    szz = [i*ones([J,1]),samples];
                    YY = tensor(Y);
                    yy(i,:) = YY(szz);
                end
                yy = rescaling .* yy';

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
                
                % compute coef matrix
                uu = tensorprod(U_right,U_right,1,1);
                vv = tensorprod(V,V,1,1);
                HTH = tensorprod(uu,vv,[1 3],[2 4]);
                HTH = permute(HTH,[1 3 2 4]);
                szz = size(HTH,1) * size(HTH,2);
                HTH = reshape(HTH,szz,szz);

                xx = reshape(X,NN,size(X,2),prod(size(X,3:L+1)));
                C = tensorprod(xx,U_right,3,1);
                cc = tensorprod(C,C,1,1);
                ATA = tensorprod(cc,vv,[2 5],[2 4]);
                ATA = permute(ATA,[1 2 5 3 4 6]);
                szz = szz * size(ATA,1);
                ATA = reshape(ATA,szz,szz);
             
                yy = yy';
                AY = tensorprod(C,yy,1,1);
                AY = tensorprod(AY,VV,[2 4],[3 1]);
                AY = reshape(AY,numel(AY),1);

                HH = lambda * kron(HTH,eye(para.P(1)));
                zz = (ATA + HH) \ (AY);
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

                G{1} = re_cores{L+1};
                for i = L+2:N
                    G{i-L} = permute(re_cores{i},[2 1 3:length(re_cores{i})]);
                end
                for i = 1:M
                    U = col(classical_mode_unfolding(G{i}, 1));
                    sampling_probs{i} = sum(abs(U).^2, 2)/size(U, 2);
                end

                samples = nan(J, M);
                barG = bSample(G,N);
                for m = 1:M
                    samples(:, m) = randsample(sz(m), J, true, sampling_probs{m});
                    GG{m} = barG{m}(samples(:,m), :, :);
                end
                core_samples = aSample(GG,G,embedding_dims,N);

                rescaling = ones(J,1);
                for m = 1:M
                    rescaling = rescaling ./ sqrt(sampling_probs{m}(samples(:, m)));
                end
                rescaling = rescaling ./ sqrt(J);

                VV = core_samples{1};
                for i = 2:M
                    VV = half_tensor_product(VV,core_samples{i});
                end

                szzz = size(VV);
                VV = classical_mode_unfolding(VV,1);
                VV = rescaling .* VV;
                VV = reshape(VV,szzz);
                
                yy = zeros(NN,J);
                for i = 1:NN
                    szz = [i*ones([J,1]),samples];
                    YY = tensor(Y);
                    yy(i,:) = YY(szz);
                end
                yy = rescaling .* yy';

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
                % compute coef matrix
                uu = tensorprod(U_left,U_left,1,1);
                vv = tensorprod(V,V,1,1);
                HTH = tensorprod(uu,vv,[2 4],[1 3]);
                HTH = permute(HTH,[1 3 2 4]);
                szz = size(HTH,1) * size(HTH,2);
                HTH = reshape(HTH,szz,szz);

                xx = reshape(X,NN,prod(size(X,2:L)),size(X,L+1));
                C = tensorprod(xx,U_left,2,1);
                cc = tensorprod(C,C,1,1);
                ATA = tensorprod(cc,vv,[3 6],[1 3]);
                ATA = permute(ATA,[1 2 5 3 4 6]);
                szz = szz * size(ATA,1);
                ATA = reshape(ATA,szz,szz);
             
                yy = yy';
                AY = tensorprod(C,yy,1,1);
                AY = tensorprod(AY,VV,[3 4],[2 1]);
                AY = reshape(AY,numel(AY),1);

                HH = lambda * kron(HTH,eye(para.P(L)));
                zz = (ATA + HH) \ (AY);
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

                G{1} = re_cores{L+1};
                for i = L+2:N
                    G{i-L} = permute(re_cores{i},[2 1 3:length(re_cores{i})]);
                end
                for i = 1:M
                    U = col(classical_mode_unfolding(G{i}, 1));
                    sampling_probs{i} = sum(abs(U).^2, 2)/size(U, 2);
                end

                samples = nan(J, M);
                barG = bSample(G,N);
                for m = 1:M
                    samples(:, m) = randsample(sz(m), J, true, sampling_probs{m});
                    GG{m} = barG{m}(samples(:,m), :, :);
                end
                core_samples = aSample(GG,G,embedding_dims,N);

                rescaling = ones(J,1);
                for m = 1:M
                    rescaling = rescaling ./ sqrt(sampling_probs{m}(samples(:, m)));
                end
                rescaling = rescaling ./ sqrt(J);

                VV = core_samples{1};
                for i = 2:M
                    VV = half_tensor_product(VV,core_samples{i});
                end

                szzz = size(VV);
                VV = classical_mode_unfolding(VV,1);
                VV = rescaling .* VV;
                VV = reshape(VV,szzz);
                
                yy = zeros(NN,J);
                for i = 1:NN
                    szz = [i*ones([J,1]),samples];
                    YY = tensor(Y);
                    yy(i,:) = YY(szz);
                end
                yy = rescaling .* yy';

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
                % compute coef matrix
                uu_left = tensorprod(U_left,U_left,1,1);
                uu_right = tensorprod(U_right,U_right,1,1);
                vv = tensorprod(V,V,1,1);
                HTH = tensorprod(uu_left,uu_right,[2 5],[2 5]);
                HTH = tensorprod(HTH,vv,[2 4 5 7],[1 4 3 6]);
                HTH = permute(HTH,[1 3 5 2 4 6]);
                szz = size(HTH,1) * size(HTH,2) * size(HTH,3);
                HTH = reshape(HTH,szz,szz);

                szzz = horzcat(NN,size(X,2:j),size(X,j+1),size(X,j+2:L+1));
                xx = reshape(X,szzz);
                C = tensorprod(xx,U_left,2,1);
                C = tensorprod(C,U_right,[3 5],[1 3]);
                cc = tensorprod(C,C,1,1);
                ATA = tensorprod(cc,vv,[3 4 8 9],[1 3 4 6]);
                ATA = permute(ATA,[1 2 3 7 4 5 6 8]);
                szz = szz * size(ATA,1);
                ATA = reshape(ATA,szz,szz);
             
                yy = yy';
                AY = tensorprod(C,yy,1,1);
                AY = tensorprod(AY,VV,[3 4 6],[2 4 1]);
                AY = reshape(AY,numel(AY),1);

                HH = lambda * kron(HTH,eye(para.P(j)));
                zz = (ATA + HH) \ (AY);
                szz = horzcat(size(cores{j},j),prod(size(cores{j},1:j-1)),prod(size(cores{j},j+1:N)));
                zz = permute(reshape(zz,szz),[2 1 3]);
                cores{j} = reshape(zz,size(cores{j}));   
            end 
        end
     end
    
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
        % compute coef matrix
        F = U;
        xx = reshape(X,NN,numel(X)/NN);
        B = xx * F;

        yy = Y';
        zz = yy*B / (B'*B + lambda*(F')*F);
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
        xx = reshape(X,NN,numel(X)/NN);

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
                % compute coef matrix
                uu = tensorprod(U,U,1,1);
                vv = tensorprod(V_right,V_right,1,1);
                FTF = tensorprod(uu,vv,[2 4],[1 3]);
                FTF = permute(FTF,[1 3 2 4]);
                szz = size(FTF,1) * size(FTF,2);
                FTF = reshape(FTF,szz,szz);

                D = tensorprod(xx,U,2,1);
                dd = tensorprod(D,D,1,1);
                BTB = tensorprod(dd,vv,[2 4],[1 3]);
                BTB = permute(BTB,[1 3 2 4]);
                BTB = reshape(BTB,szz,szz);

                szz = horzcat(NN,size(Y,2),prod(size(Y,3:M+1)));
                y = reshape(Y,szz);
                YB = tensorprod(y,D,1,1);
                YB = tensorprod(YB,V_right,[2 4],[1 2]);
                YB = reshape(YB,size(YB,1),numel(YB)/size(YB,1));

                zz = YB / (BTB + lambda*FTF);
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
                V_right = re_cores{L+1};
                for i = L+2:N-1
                    V_right = tensor_prod(V_right,re_cores{i});
                end
                szz = horzcat(size(UU,1),prod(size(UU,2:M)),size(UU,M+1));
                U = reshape(UU,szz);
                % compute coef matrix
                uu = tensorprod(U,U,1,1);
                vv = tensorprod(V_right,V_right,1,1);
                FTF = tensorprod(uu,vv,[1 3],[2 4]);
                FTF = permute(FTF,[1 3 2 4]);
                szz = size(FTF,1) * size(FTF,2);
                FTF = reshape(FTF,szz,szz);

                D = tensorprod(xx,U,2,1);
                dd = tensorprod(D,D,1,1);
                BTB = tensorprod(dd,vv,[1 3],[2 4]);
                BTB = permute(BTB,[1 3 2 4]);
                BTB = reshape(BTB,szz,szz);

                szz = horzcat(NN,prod(size(Y,2:M)),size(Y,M+1));
                y = reshape(Y,szz);
                YB = tensorprod(y,D,1,1);
                YB = tensorprod(YB,V_right,[1 3],[1 3]);
                YB = reshape(YB,size(YB,1),numel(YB)/size(YB,1));

                zz = YB / (BTB + lambda*FTF);
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
                % compute coef matrix
                uu = tensorprod(U,U,1,1);
                vv_left = tensorprod(V_left,V_left,1,1);
                vv_right = tensorprod(V_right,V_right,1,1);
                FTF = tensorprod(uu,vv_left,[1 4],[3 6]);
                FTF = tensorprod(FTF,vv_right,[2 4 6 8],[1 4 2 5]);
                FTF = permute(FTF,[1 3 5 2 4 6]);
                szz = size(FTF,1) * size(FTF,2) * size(FTF,3);
                FTF = reshape(FTF,szz,szz);

                D = tensorprod(xx,U,2,1);
                dd = tensorprod(D,D,1,1);
                BTB = tensorprod(dd,vv_left,[1 4],[3 6]);
                BTB = tensorprod(BTB,vv_right,[2 4 6 8],[1 4 2 5]);
                BTB = permute(BTB,[1 3 5 2 4 6]);
                BTB = reshape(BTB,szz,szz);

                szz = horzcat(NN,prod(size(Y,2:k)),size(Y,k+1),prod(size(Y,k+2:M+1)));
                y = reshape(Y,szz);
                YB = tensorprod(y,D,1,1);
                YB = tensorprod(YB,V_left,[1 4],[1 4]);
                YB = tensorprod(YB,V_right,[2 4 6],[1 2 3]);
                YB = reshape(YB,size(YB,1),numel(YB)/size(YB,1));

                zz = YB / (BTB + lambda*FTF);
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
        best_runtime = runtime;
        best_valerror = error;
    end
end