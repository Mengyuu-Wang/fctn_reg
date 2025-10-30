addpath("data1\")
load("USHCN_testx.mat")
load("USHCN_testy.mat")
load("USHCN_trainx.mat")
load("USHCN_trainy.mat")
load("USHCN_valx.mat")
load("USHCN_valy.mat")
para.P = [23 3 3];
para.Q = [23 3];
para.L = length(para.P);
para.M = length(para.Q);
para.dim = [para.P,para.Q];
para.N = length(para.dim);
para.maxiter=1000;
para.datarep=20;
para.lambda = 10^(3);

           
%% main part
addpath("fctn\")
rank = [0 2 2 2 2 
        0 0 2 2 2 
        0 0 0 2 2
        0 0 0 0 2
        0 0 0 0 0];
[est_model,runtime_ttr] = fctn_reg_str(para, rank, trainx, trainy, valx, valy);

% addpath("tt\")
% rank = [1,14,9,7,3,1];
% [est_model,runtime_ttr] = ttr(para, rank, trainx, trainy, valx, valy);

% addpath("tr\")
% rank = [7,7,5,6,6];
% [est_model,runtime_ttr] = tr_reg(para, rank, trainx, trainy, valx, valy);

% est_testy and error
est_testy=contract(testx,est_model,para.L);
cor_ttr = mycorrcoef(est_testy(:),testy(:));
RMSE = sqrt(norm(est_testy(:)-testy(:))^2  / length(est_testy(:)));
NRMSE = RMSE / mean(testy(:));
MAE = sum(abs(est_testy(:)-testy(:)))/length(est_testy(:));
NMAE = MAE / mean(testy(:));
Q2_ttr = 1 - norm(testy(:)-est_testy(:),"fro")^2/norm(testy(:),"fro")^2;



