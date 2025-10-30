%%%%%%%%%%%%%%%%%%%%%%%%
addpath("fctn_als\")
rank_gen = [0,2,3,4
            0,0,2,3
            0,0,0,4
            0,0,0,0];
sz = [20,20,20,20];
X_true = gengengen(sz,rank_gen,0,0);
%X = X_true + 0.01*randn(sz);
X = X_true + 0.01*rand(2,sz);

best_error = Inf;
RESULT=[];
gap = 2:4;
paramGrid = struct('R12', gap, 'R13', gap,'R14', gap, ...
    'R23', gap,'R24', gap,'R34', gap);
for a = 1:length(paramGrid.R12)
    for b = 1:length(paramGrid.R13)
        for c = 1:length(paramGrid.R14)
            for d = 1:length(paramGrid.R23)
                for e = 1:length(paramGrid.R24)
                    for f = 1:length(paramGrid.R34)
                        params = struct('R12', paramGrid.R12(a),'R13', ...
                            paramGrid.R13(b),'R14', paramGrid.R14(c),'R23', ...
                            paramGrid.R23(d),'R24', paramGrid.R24(e),'R34', ...
                            paramGrid.R34(f));
                        rank=[0 params.R12(1) params.R13(1) params.R14(1)
                              0 0 params.R23(1) params.R24(1)
                              0 0 0 params.R34(1)
                              0 0 0 0];
                        %% FCTN-ALS
                        [cores,~,~] = FCTN_ALS(X,rank,"tol",1e-6,"maxiters",50);
                        Y = cores_2_tensor(cores,sz);
                        error = norm(Y(:)-X_true(:))/norm(X_true(:));

                        result=struct('para',params,'error',error);
                        RESULT=[RESULT,result];
                        % 如果性能更好，则更新最佳参数
                        if  error < best_error
                            best_error = error;
                            bestParams = params;
                        end
                    end
                end
            end
        end
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%
addpath("fctn_als\")
best_error = Inf;
RESULT=[];
gap = 2:4;
fixed = 4;
paramGrid = struct('R12', gap, 'R13', gap,'R14', gap, ...
    'R23', fixed,'R24', fixed,'R34', fixed);
for a = 1:length(paramGrid.R12)
    for b = 1:length(paramGrid.R13)
        for c = 1:length(paramGrid.R14)
            for d = 1:length(paramGrid.R23)
                for e = 1:length(paramGrid.R24)
                    for f = 1:length(paramGrid.R34)
                        params = struct('R12', paramGrid.R12(a),'R13', ...
                            paramGrid.R13(b),'R14', paramGrid.R14(c),'R23', ...
                            paramGrid.R23(d),'R24', paramGrid.R24(e),'R34', ...
                            paramGrid.R34(f));
                        rank=[0 params.R12(1) params.R13(1) params.R14(1)
                              0 0 params.R23(1) params.R24(1)
                              0 0 0 params.R34(1)
                              0 0 0 0];
                        %% FCTN-ALS
                        [cores,~,~] = FCTN_ALS(X,rank,"tol",1e-6,"maxiters",50);
                        Y = cores_2_tensor(cores,sz);
                        error = norm(Y(:)-X_true(:))/norm(X_true(:));

                        result=struct('para',params,'error',error);
                        RESULT=[RESULT,result];
                        % 如果性能更好，则更新最佳参数
                        if  error < best_error
                            best_error = error;
                            bestParams = params;
                        end
                    end
                end
            end
        end
    end
end



%%%%%%%%%%%%%%%%%%%%%%%%
addpath("fctn_als\")
best_error = Inf;
RESULT=[];
gap = 2:4;
paramGrid = struct('R12', bestParams.R12, 'R13', bestParams.R13,'R14', bestParams.R14, ...
    'R23', gap,'R24', gap,'R34', gap);
for a = 1:length(paramGrid.R12)
    for b = 1:length(paramGrid.R13)
        for c = 1:length(paramGrid.R14)
            for d = 1:length(paramGrid.R23)
                for e = 1:length(paramGrid.R24)
                    for f = 1:length(paramGrid.R34)
                        params = struct('R12', paramGrid.R12(a),'R13', ...
                            paramGrid.R13(b),'R14', paramGrid.R14(c),'R23', ...
                            paramGrid.R23(d),'R24', paramGrid.R24(e),'R34', ...
                            paramGrid.R34(f));
                        rank=[0 params.R12(1) params.R13(1) params.R14(1)
                              0 0 params.R23(1) params.R24(1)
                              0 0 0 params.R34(1)
                              0 0 0 0];
                        %% FCTN-ALS
                        [cores,~,~] = FCTN_ALS(X,rank,"tol",1e-6,"maxiters",50);
                        Y = cores_2_tensor(cores,sz);
                        error = norm(Y(:)-X_true(:))/norm(X_true(:));

                        result=struct('para',params,'error',error);
                        RESULT=[RESULT,result];
                        % 如果性能更好，则更新最佳参数
                        if  error < best_error
                            best_error = error;
                            bestParams = params;
                        end
                    end
                end
            end
        end
    end
end
