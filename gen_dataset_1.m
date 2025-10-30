clc; clear
addpath("fctn\")
% 参数设置
para.P = [15, 20];
para.Q = [5, 10];
para.L = length(para.P);
para.M = length(para.Q);
para.dim = [para.P, para.Q];
para.maxiter = 1000;
para.datarep = 20;

% 样本数量列表
samplelist = {60:20:240};  

% rank 设置
ranks = {[0 2 2 2;
          0 0 2 2;
          0 0 0 2; 
          0 0 0 0]};
% 创建数据保存文件夹
datapath = fullfile(pwd, 'data1');
if ~exist(datapath, 'dir')
    mkdir(datapath);
end

% 开始循环生成数据
for s = 1:2
    s2n = 5 * s;
    noiselevel = 1 / s2n;

    for R_ind = 1:2
        for n_ind = 1:10
                n = samplelist{R_ind}{1}(n_ind);
            for i = 1
                % 设置样本大小
                para.N = n;

                % 生成模型
                model = init_model(para, ranks{R_ind});

                % 生成 train/val/test 数据
                trainx = randn([para.N, para.P]);
                trainy = contract(trainx, model, para.L) + noiselevel * randn([para.N, para.Q]);

                valx = randn([para.N, para.P]);
                valy = contract(valx, model, para.L) + noiselevel * randn([para.N, para.Q]);

                testx = randn([para.N, para.P]);
                testy = contract(testx, model, para.L) + noiselevel * randn([para.N, para.Q]);

                % 构造文件名前缀（不含路径）
                prefix = ['_', num2str(R_ind), '_', num2str(s2n), '_', num2str(n), '_', num2str(i)];
                
                % 保存数据
                save(fullfile(datapath, ['trainx', prefix, '.mat']), 'trainx');
                save(fullfile(datapath, ['trainy', prefix, '.mat']), 'trainy');
                save(fullfile(datapath, ['valx',   prefix, '.mat']), 'valx');
                save(fullfile(datapath, ['valy',   prefix, '.mat']), 'valy');
                save(fullfile(datapath, ['testx',  prefix, '.mat']), 'testx');
                save(fullfile(datapath, ['testy',  prefix, '.mat']), 'testy');
            end
        end
    end
end
