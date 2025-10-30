clc; clear;

%% 参数设置
para.P = [15, 20];
para.Q = [5, 10];
para.L = length(para.P);
para.M = length(para.Q);
para.dim = [para.P,para.Q];
para.maxiter = 1000;
para.lambda = 10^(-2);

% rank 配置
tt_rank_list = {
    [1, 8, 16, 10, 1];       
    [1, 15, 20, 11, 1];      
};
tr_rank_list = {
    [5, 4, 4, 6];
    [7, 7, 6, 6]
};
fctn_rank_list = {
    [0, 2, 2, 2
     0, 0, 2, 2
     0, 0, 0, 2
     0, 0, 0, 0];
    [0, 2, 3, 2
     0, 0, 3, 2
     0, 0, 0, 3
     0, 0, 0, 0]
};
% 样本量设置
samplelist = {
    100:20:280;              
    140:20:320             
};

datapath = './data/';
reps = 10;

%% 初始化结果矩阵
RMSEmat_tt = zeros(4, 10);   % 4组 × 10个样本量
TIMEmat_tt = zeros(4, 10);
RMSEmat_tr  = zeros(4, 10);
TIMEmat_tr  = zeros(4, 10);
RMSEmat_fctn  = zeros(4, 10);
TIMEmat_fctn  = zeros(4, 10);

%% 实验开始
for s = 1:2  % SNR level: 10, 20
    s2n = 10 * s;

    for R_ind = 1:2  % rank 组
        fprintf('\n=== Running: SNR=%d, R_ind=%d ===\n', s2n, R_ind);
        rk_tt = tt_rank_list{R_ind};
        rk_tr = tr_rank_list{R_ind};
        rk_fctn = fctn_rank_list{R_ind};
        ns = samplelist{R_ind};

        group_id = (s - 1) * 2 + R_ind;

        for n_ind = 1:length(ns)
            n = ns(n_ind);

            rmse_list_tt = zeros(reps, 1);
            runtime_list_tt = zeros(reps, 1);
            rmse_list_tr = zeros(reps, 1);
            runtime_list_tr = zeros(reps, 1);
            rmse_list_fctn = zeros(reps, 1);
            runtime_list_fctn = zeros(reps, 1);

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

                %% run tt
                addpath("tt\tensor_toolbox-v3.2\","tt\TT-Toolbox-master\TT-Toolbox-master\core\")
                addpath("tt\tools\","tt\demo\","tt\TT-Toolbox-master\TT-Toolbox-master\")
                para.N = n;
                [model, runtime] = ttr(para, rk_tt, trainx, trainy, valx, valy);
                est_model = full(model);
                est_model = reshape(est_model,size(model));

                % 测试预测
                est_testy = contract(testx, est_model, para.L);
                Ypred = zscore(est_testy(:));
                Ytrue = zscore(testy(:));
                rmse_val = sqrt(mean((Ytrue - Ypred).^2));      

                % 记录
                rmse_list_tt(i) = rmse_val;
                runtime_list_tt(i) = runtime;

                %% run tr
                addpath("tr\")
                para.N = length(para.dim);
                [model, runtime] = tr_reg(para, rk_tr, trainx, trainy, valx, valy);
                est_testy_tr = contract(testx, model, para.L);
                Ypred = zscore(est_testy_tr(:));

                rmse_list_tr(i) = sqrt(mean((Ypred - Ytrue).^2));
                runtime_list_tr(i) = runtime;

                %% run fctn_fast
                addpath("fctn\","fctn\mtimesx\mtimesx_20110223\")
                [model, runtime] = fctn_reg_str(para, rk_fctn, trainx, trainy, valx, valy);
                est_testy_fctn = contract(testx, model, para.L);
                Ypred = zscore(est_testy_fctn(:));

                rmse_list_fctn(i) = sqrt(mean((Ypred - Ytrue).^2));
                runtime_list_fctn(i) = runtime;
            end

            % 平均记录
            RMSEmat_tt(group_id, n_ind) = mean(rmse_list_tt);
            TIMEmat_tt(group_id, n_ind) = mean(runtime_list_tt);
            RMSEmat_tr(group_id, n_ind)  = mean(rmse_list_tr);
            TIMEmat_tr(group_id, n_ind)  = mean(runtime_list_tr);
            RMSEmat_fctn(group_id, n_ind)  = mean(rmse_list_fctn);
            TIMEmat_fctn(group_id, n_ind)  = mean(runtime_list_fctn);
            
        end
    end
end


