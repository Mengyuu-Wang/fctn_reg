function [model,runtime,trainerror ] = fctn_reg( para, rank, X,Y, Xv, Yv )
%% set parameters

iterStart = 0;
iterTotal = para.maxiter;
shakyRate = 1.5;
N=para.N;
L=para.L;
M=para.M;
lambda=para.lambda;
NN = size(X,1);

%% initialization
%initialize the random model
[model,cores] = init_model(para,rank);
re_cores = cell(N,1);
%% 停止条件准备
minTrainingFuncValue = calcobjfunction(para,model, X,Y);
minValidationFuncValue = calcobjfunction(para,model,Xv,Yv);

disp('Train the model. Running...');
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
        H = re_cores{2};
        for i = 3:N
            H = tensor_prod(H,re_cores{i});
        end
        A = kron(X,H);
        yy = reshape(Y,numel(Y),1);
        HH = lambda * kron((H')*H,eye(para.P(1)));
        zz = (A'*A + HH) \ (A'*yy);
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
                % compute coef matrix
                H = tensorprod(U_right,V,2,3);
                H = permute(H,[1 3 2 4]); 
                xx = reshape(X,NN,size(X,2),prod(size(X,3:L+1)));
                A = tensorprod(xx,H,3,1);
                A = permute(A,[1 3 2 4 5]);
                H = reshape(H,size(H,1)*size(H,2),size(H,3)*size(H,4));
                A = reshape(A,size(A,1)*size(A,2),numel(A)/(size(A,1)*size(A,2)));
                
                yy = reshape(Y,numel(Y),1);
                HH = lambda * kron((H')*H,eye(para.P(1)));
                zz = (A'*A + HH) \ (A'*yy);
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
                % compute coef matrix
                H = tensorprod(U_left,V,3,2);
                H = permute(H,[1 3 2 4]); 
                xx = reshape(X,NN,prod(size(X,2:L)),size(X,L+1));
                A = tensorprod(xx,H,2,1);
                A = permute(A,[1 3 2 4 5]);
                H = reshape(H,size(H,1)*size(H,2),size(H,3)*size(H,4));
                A = reshape(A,size(A,1)*size(A,2),numel(A)/(size(A,1)*size(A,2)));
                
                yy = reshape(Y,numel(Y),1);
                HH = lambda * kron((H')*H,eye(para.P(L)));
                zz = (A'*A + HH) \ (A'*yy);
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
                % compute coef matrix
                H = tensorprod(U_left,U_right,3,3);
                H = tensorprod(H,V,[3 5],[2 4]);
                szz = horzcat(NN,size(X,2:j),size(X,j+1),size(X,j+2:L+1));
                xx = reshape(X,szz);
                A = tensorprod(xx,H,[2 4],[1 3]);
                A = permute(A,[1 5 2 3 4 6]);
                H = permute(H,[1 3 5 2 4 6]);
                ee = NN*prod(para.Q);
                A = reshape(A,ee,numel(A)/ee);
                ee = prod(para.dim)/para.P(j);
                H = reshape(H,ee,numel(H)/ee);

                yy = reshape(Y,numel(Y),1);
                HH = lambda * kron((H')*H,eye(para.P(j)));
                zz = (A'*A + HH) \ (A'*yy);
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
        % compute coef matrix
        F = U;
        xx = reshape(X,NN,numel(X)/NN);
        B = xx * F;

        yy = Y';
        zz = (yy*B) * inv(B'*B + lambda*(F')*F);
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
                F = tensorprod(U,V_right,3,2);
                xx = reshape(X,NN,numel(X)/NN);
                F = permute(F,[1 3 2 4]);
                B = tensorprod(xx,F,2,1);
                F = reshape(F,size(F,1)*size(F,2),size(F,3)*size(F,4));
                B = reshape(B,size(B,1)*size(B,2),size(B,3)*size(B,4));

                yy = classical_mode_unfolding(Y,1+k);
                zz = (yy*B) * inv(B'*B + lambda*(F')*F);
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
                F = tensorprod(U,V_right,2,3);
                xx = reshape(X,NN,numel(X)/NN);
                F = permute(F,[1 3 2 4]);
                B = tensorprod(xx,F,2,1);
                F = reshape(F,size(F,1)*size(F,2),size(F,3)*size(F,4));
                B = reshape(B,size(B,1)*size(B,2),size(B,3)*size(B,4));

                yy = classical_mode_unfolding(Y,1+k);
                zz = (yy*B) * inv(B'*B + lambda*(F')*F);
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
                F = tensorprod(U,V_left,2,4);
                F = tensorprod(F,V_right,[3 6],[2 3]);
                xx = reshape(X,NN,numel(X)/NN);
                F = permute(F,[1 3 5 2 4 6]);
                B = tensorprod(xx,F,2,1);
                F = reshape(F,size(F,1)*size(F,2)*size(F,3),size(F,4)*size(F,5)*size(F,6));
                B = reshape(B,size(B,1)*size(B,2)*size(B,3),size(B,4)*size(B,5)*size(B,6));

                yy = classical_mode_unfolding(Y,1+k);
                zz = (yy*B) * inv(B'*B + lambda*(F')*F);
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
runtime=toc;
disp('Train the model. Finish.');

end