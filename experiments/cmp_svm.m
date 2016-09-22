diary('cmp_svm.log');

system('python tsnet/datasets.py');
datasets = {'mnist' 'cifar10' 'svhn2'};

clip   = @(K)max(min(K,1),-1);
ssrelu = @(K)(sqrt(1-K.^2) + (pi-acos(K)).*K)/pi;
tsrelu = @(K)(1 - acos(K)/pi).*K;

kern = { ssrelu ,  tsrelu };
name = {'SSReLU', 'TSReLU'};

for i = 1:length(datasets)
    
    fprintf([datasets{i} '\n']);
    load(datasets{i});
    
    X_trn = reshape(X_trn, size(X_trn,1), []);
    X_val = reshape(X_val, size(X_val,1), []);
    X_tst = reshape(X_tst, size(X_tst,1), []);
    
    X_trn = bsxfun(@rdivide, X_trn, sqrt(sum(X_trn.^2, 2)));
    X_val = bsxfun(@rdivide, X_val, sqrt(sum(X_val.^2, 2)));
    X_tst = bsxfun(@rdivide, X_tst, sqrt(sum(X_tst.^2, 2)));
    
    y_trn = double(y_trn);
    y_val = double(y_val);
    y_tst = double(y_tst);
    
    for k = [1 2]
        for c = 10 .^ [0 1 2 3]
            
            K_trn = zeros(size(X_trn,1)+1, size(X_trn,1), 'single'); K_trn(1,:) = 1:size(X_trn,1);
            K_val = zeros(size(X_trn,1)+1, size(X_val,1), 'single'); K_val(1,:) = 1:size(X_val,1);
            K_tst = zeros(size(X_trn,1)+1, size(X_tst,1), 'single'); K_tst(1,:) = 1:size(X_tst,1);
            
            tic;
            K_trn(2:end,:) = clip(X_trn * X_trn');
            K_val(2:end,:) = clip(X_trn * X_val'); t_base = toc;
            K_tst(2:end,:) = clip(X_trn * X_tst');
            
            acc = 0;
            for l = 0:100
                
                fprintf('%s: c=%.1e, l=%02d, ', name{k}, c, l);
                
                tic;
                model = svmtrain_inplace(y_trn, K_trn, sprintf('-t 4 -c %d -q', c));
                fprintf('nSV=%d, ', model.totalSV);
                
                [~, tmp, ~]    = svmpredict_inplace(y_val, K_val, model); t = toc;
                fprintf('\b, '); svmpredict_inplace(y_tst, K_tst, model);
                fprintf('\b, t=%.2f\n', t_base+t);
                
                if tmp(1) >= acc, acc = tmp(1); else, break; end
                
                tic;
                K_trn(2:end,:) = clip(kern{k}(K_trn(2:end,:)));
                K_val(2:end,:) = clip(kern{k}(K_val(2:end,:))); t_base = t_base + toc;
                K_tst(2:end,:) = clip(kern{k}(K_tst(2:end,:)));
            end
        end
    end
end

diary off;