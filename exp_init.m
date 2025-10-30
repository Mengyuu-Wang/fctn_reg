clc; clear;

%% 参数设置
para.P = [20, 20];
para.Q = [20, 20];
para.L = length(para.P);
para.M = length(para.Q);
para.dim = [para.P,para.Q];
para.maxiter = 1000;
para.N = length(para.dim);
para.lambda = 10^(-2);

% rank 配置
fctn_rank_list = {
    [0, 2, 2, 2
     0, 0, 2, 2
     0, 0, 0, 2
     0, 0, 0, 0]
};
% 样本量设置
samplelist = {
    60:20:240            
};

datapath = './data_init/';
reps = 10;

%% 初始化结果矩阵
RMSEmat_fctn_random  = zeros(4, 10);
TIMEmat_fctn_random  = zeros(4, 10);
RMSEmat_fctn_svd  = zeros(4, 10);
TIMEmat_fctn_svd  = zeros(4, 10);

%% 实验开始
for s = 1  
    s2n = 5 * s;

    for R_ind = 1  % rank 组
        fprintf('\n=== Running: SNR=%d, R_ind=%d ===\n', s2n, R_ind);
        rk_fctn = fctn_rank_list{R_ind};
        ns = samplelist{R_ind};

        group_id = (s - 1) * 2 + R_ind;

        for n_ind = 1:length(ns)
            n = ns(n_ind);

            rmse_list_fctnrandom = zeros(reps, 1);
            runtime_list_fctnrandom = zeros(reps, 1);
            rmse_list_fctnsvd = zeros(reps, 1);
            runtime_list_fctnsvd  = zeros(reps, 1);

            for i = 1:reps
                suffix = ['_', num2str(R_ind), '_', num2str(s2n), '_', num2str(n), '_', num2str(1)];

                try
                    load(fullfile(datapath, ['trainx', suffix, '.mat']), 'trainx');
                    load(fullfile(datapath, ['trainy', suffix, '.mat']), 'trainy');
                    load(fullfile(datapath, ['valx',   suffix, '.mat']), 'valx');
                    load(fullfile(datapath, ['valy',   suffix, '.mat']), 'valy');
                    load(fullfile(datapath, ['testx',  suffix, '.mat']), 'testx');
                    load(fullfile(datapath, ['testy',  suffix, '.mat']), 'testy');
                catch
                    warning('数据 %s 缺失，跳过', suffix);
                    continue;
                end

                % 标准化
                standardize = @(x) (x - mean(x(:))) / std(x(:));
                trainx = standardize(trainx); trainy = standardize(trainy);
                valx   = standardize(valx);   valy   = standardize(valy);
                testx  = standardize(testx);  testy  = standardize(testy);
  
                %% run fctn_random_init
                addpath("fctn\")
                init = 1;
                [model, runtime] = fctn_reg_init(para, rk_fctn, trainx, trainy, valx, valy, init);
                est_testy_fctn = contract(testx, model, para.L);
                Ypred = zscore(est_testy_fctn(:));
                Ytrue = zscore(testy(:));

                rmse_list_fctnrandom(i) = sqrt(mean((Ypred - Ytrue).^2));
                runtime_list_fctnrandom(i) = runtime;
                
                %% run fctn_random_svd
                init = 0;
                [model, runtime] = fctn_reg_init(para, rk_fctn, trainx, trainy, valx, valy, init);
                est_testy_fctn = contract(testx, model, para.L);
                Ypred = zscore(est_testy_fctn(:));
                
                rmse_list_fctnsvd(i) = sqrt(mean((Ypred - Ytrue).^2));
                runtime_list_fctnsvd(i) = runtime;
            end

            RMSEmat_fctn_random(group_id, n_ind)  = mean(rmse_list_fctnrandom);
            TIMEmat_fctn_random(group_id, n_ind)  = mean(runtime_list_fctnrandom);
            RMSEmat_fctn_svd(group_id, n_ind)  = mean(rmse_list_fctnsvd);
            TIMEmat_fctn_svd(group_id, n_ind)  = mean(runtime_list_fctnsvd);
            
        end
    end
end


