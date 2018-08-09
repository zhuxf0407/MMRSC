function Run_Experiment_MMRSC(dataset, maxiter, numRuns)
%Usage: Run_Experiment_MMRSC('lastfm',30,20)
%       Run_Experiment_MMRSC('mirflickr',15,20)
%       Run_Experiment_MMRSC('handwritten',15,20) 

  
addpath('GraphSC');
addpath('./JialuLIU/Code_multiNMF/mulNMF');
speak = 0;

model = 'MMRSC';
%maxiter = 50;
%dataset = 'lastfm';
switch lower(dataset)
    case 'mirflickr'
        %% load data matrix
        disp('Using dataset mirflickr');        
        if(computer == 'PCWIN64'),
            load D:\academic\Dataset\SparseCodingExperiments\Experiments\Exp00_PrepareDataset\Clustering\MIRFLICKR\mirflickr.mat
        else
            load /data3/zhu/SparseCodingExperiments/Exp00_PrepareDataset/Clustering/MIRFLICKR/mirflickr.mat;
        end
        X = cell(2,1);  X{1} = fea_visual; X{2} = fea_tag;
        nCluster = 10; K = nCluster;
        label = gnd;   
  
    case 'lastfm'
        disp('Using dataset lastfm')
        if(computer == 'PCWIN64'),
            load D:\academic\Dataset\SparseCodingExperiments\Experiments\Exp00_PrepareDataset\Clustering\Lastfm\processed\lastfm_tfidf.mat;
        else
            load /data3/zhu/SparseCodingExperiments/Exp01_LearnSparseCodingRepresentation/Clustering/Lastfm/processed/lastfm_tfidf.mat;
        end
        % gnd: 9694 x 1, commwords: 9694 x 31172,  descwords: 9694 x 14076, users: 9694 x 131353
        X = cell(3,1); X{1} = commwords; X{2} = descwords; X{3} = users;  clear commwords descwords users;
        nCluster = 21; K = nCluster;
        label = gnd;
        
    case 'handwritten'
        disp('Using dataset handwritten')
        load ./JialuLIU/Code_multiNMF/handwritten.mat;
        X = cell(2,1); X{1} = fourier; X{2} = pixel;  clear commwords fourier pixel;         
        nCluster = 10; K = nCluster;
        label = gnd;   
    case '3source'
        disp('Using dataset 3source')
        load ./3source.mat;
        X = cell(3,1); X{1} = bbc; X{2} = guardian; X{3} = reuters; clear bbc guardian reuters;         
        nCluster = 6; K = nCluster;
        label = gnd;    
    otherwise
        error('Unknown dataset. Available datasets are mirflickr or lastfm.');
        return
end

%% only for test: use subset to accelerate speed
% % nView = length(X); 
% % num = size(X{1},1);
% % rp = randperm(num);
% % Idx = rp(1:1000);
% % for i=1:length(X),
% %     X{i} = X{i}(Idx,:);
% % end
% % label = label(Idx);
% % -----------------------------------------------------


%% =============================================================================
disp('Running MMRSC ...'); 
%% 1 test for negative values in X
nView = length(X); 
for i = 1: nView,
    if min(min(X{i})) < 0
        error('Input matrix X{%d} elements can not be negative',i);
        return
    end
end

%% Normalization Step
% % for i = 1:length(X)
% %     X{i} = X{i} / sum(sum(X{i})); 
% % end
% % for i = 1:length(X)
% %     X{i} = X{i} / max(max(X{i}));  
% % end

Y = [X{:}];  
%% 2. Using GraphSC on each view   
options = []; 
options.NeighborMode = 'KNN';
options.k = 3;% options.k=10;
options.WeightMode = 'HeatKernel';
options.t = 1;  
W = cell(nView,1);
for i=1:nView, 
    nSam = size(X{i},1); 
    maxSam = nSam; 
    if(maxSam > 1000),
        maxSam = 1000;
    end
    idx = randperm(nSam);
    tmpX = X{i}(idx(1:maxSam),:);
    options.t = sqrt(mean(mean(pdist2(tmpX,tmpX,'euclidean')))); 
    W{i} = constructW(X{i},options);
end

V=zeros(size(W{1}));
for i=1:nView,
    V = V + W{i}; 
end

sname = sprintf('data_normalize_no_%s_%s_%d.mat',model,dataset,maxiter);
save(sname);
disp('ok'); 

% % sname = sprintf('data_normalize_no_%s_%s_%d.mat',model,dataset,maxiter);
% % load(sname);
% % ===============================end==========================================

sname = sprintf('log_%s_%s_%d.txt',model,dataset,maxiter); 
flog = fopen(sname,'a');
fprintf(flog,'\n\n%s:\n',datestr(now)  );

%% ConcateMMRSC 

nBasis = 64;% K;% K; %gen newfea
alpha = 1;%1; %1;
beta = 0.1; %0.1;%*ones(1,nView); %0.1;
nIters = maxiter; 
warning('off', 'all');

fprintf(flog, 'Run\tAC\tNMI\n'); fprintf('Run\tAC\tNMI\n');
AC = zeros(1,numRuns);
NMI = zeros(1,numRuns); 

for i = 1:numRuns,  
    %rand('twister', sum(100*clock));
    %rand('twister',5489); 
    tic 
    [B, S, stat] = GraphSC(flog, Y', V, nBasis, alpha, beta,K,label, nIters); 
    
    predict = litekmeans(S', K, 'Replicates',30);   %using the same initialization of kmeans for fair comparison.
    
    [AC(i), NMI(i), cnt] = CalcMetrics(label, predict); 
    time = toc;
    fprintf(flog,'%d\t%.3f\t%.3f\t%.3f\n',i,AC(i),NMI(i),time); 
    fprintf('%d\t%.3f\t%.3f\t%.3f\n',i,AC(i),NMI(i),time); 
    
end

%% Output results
AC_avg = mean(AC); AC_std = std(AC); NMI_avg = mean(NMI); NMI_std = std(NMI); 
fprintf(flog,'ALL\t%.3f(%.3f)\t%.3f(%.3f)\n',AC_avg,AC_std,NMI_avg,NMI_std);
fprintf('ALL\t%.3f(%.3f)\t%.3f(%.3f)\n',AC_avg,AC_std,NMI_avg,NMI_std);
 

fclose(flog);


rmpath('./JialuLIU/Code_multiNMF/mulNMF');
rmpath('GraphSC'); 

