function cmp_svm()

diary('cmp_svm.log');

%system('python tsnet/datasets.py');
datasets = {'mnist' 'cifar10' 'svhn2'};
L = 10;

clip   = @(K)      max(min(K,1),-1);
ssrelu = @(Kss)    (sqrt(1-Kss.^2) + (pi-acos(Kss)).*Kss)/pi;
tsrelu = @(Kss,Kts)(1 - acos(Kss)/pi).*Kts;

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
    
    for c = [0 1 2 3]
        
        Kss_trn = zeros(size(X_trn,1)+1, size(X_trn,1), 'single'); Kss_trn(1,:) = 1:size(X_trn,1);
        Kss_val = zeros(size(X_trn,1)+1, size(X_val,1), 'single'); Kss_val(1,:) = 1:size(X_val,1);
        Kss_tst = zeros(size(X_trn,1)+1, size(X_tst,1), 'single'); Kss_tst(1,:) = 1:size(X_tst,1);
        
        Rss = zeros(L+1,3);
        
        tic;
        Kss_trn(2:end,:) = clip(X_trn * X_trn');
        Kss_val(2:end,:) = clip(X_trn * X_val'); Rss(:,3) = toc;
        Kss_tst(2:end,:) = clip(X_trn * X_tst');
        
        Kts_trn = Kss_trn;
        Kts_val = Kss_val;
        Kts_tst = Kss_tst;
        Rts     = Rss;
        
        for l = 0:L
            for k = [2 1]
                
                if k == 1, fprintf('%s: c=%d, l=%02d, ', 'SSReLU', c, l);
                else,      fprintf('%s: c=%d, l=%02d, ', 'TSReLU', c, l); end
                
                if k == 1, [Rss(l+1,1), Rss(l+1,2), t] = svm(Kss_trn, y_trn, Kss_val, y_val, Kss_tst, y_tst, c);
                else,      [Rts(l+1,1), Rts(l+1,2), t] = svm(Kts_trn, y_trn, Kts_val, y_val, Kts_tst, y_tst, c); end
                
                if k == 1, Rss(l+1,3) = Rss(l+1,3) + t;
                else,      Rts(l+1,3) = Rts(l+1,3) + t; end
                
                if l == L, continue; end
                
                if k == 1
                    tic;
                    Kss_trn(2:end,:) = clip(ssrelu(Kss_trn(2:end,:)));
                    Kss_val(2:end,:) = clip(ssrelu(Kss_val(2:end,:))); t = toc;
                    Kss_tst(2:end,:) = clip(ssrelu(Kss_tst(2:end,:)));
                    
                    Rss((l+2):end,3) = Rss((l+2):end,3) + t;
                    Rts((l+2):end,3) = Rts((l+2):end,3) + t;
                else
                    tic;
                    Kts_trn(2:end,:) = clip(tsrelu(Kss_trn(2:end,:),Kts_trn(2:end,:)));
                    Kts_val(2:end,:) = clip(tsrelu(Kss_val(2:end,:),Kts_val(2:end,:))); t = toc;
                    Kts_tst(2:end,:) = clip(tsrelu(Kss_tst(2:end,:),Kts_tst(2:end,:)));
                    
                    Rts((l+2):end,3) = Rts((l+2):end,3) + t;
                end
            end
        end
        
        save([datasets{i} '-' sprintf('%d', c) '.mat'], 'Rss', 'Rts');
    end
end

diary off;

end

function [val_acc, tst_acc, t] = svm(K_trn, y_trn, K_val, y_val, K_tst, y_tst, c)

tic;
model = svmtrain_inplace(y_trn, K_trn, sprintf('-t 4 -c %d -q', 10^c));
fprintf('nSV=%d, ', model.totalSV);

[~, val_acc, ~] = svmpredict_inplace(y_val, K_val, model); val_acc = val_acc(1);
t = toc; fprintf('\b, ');
[~, tst_acc, ~] = svmpredict_inplace(y_tst, K_tst, model); tst_acc = tst_acc(1);

end