folder = 'mlp/*/';
D = {'mnist' 'cifar10' 'svhn2'};

df = [];

for d = 1:length(D)
    
    F0 = sort(strsplit(ls([folder D{d} '-1*.mat'])));
    F1 = sort(strsplit(ls([folder D{d} '-2*.mat'])));
    
    for f = 1:length(F0)
        
        if isempty(F0{f}), continue; end
        if isempty(F1{f}), continue; end
        
        load(F0{f});
        
        f0_t1 = tst_acc(1 ); [~, bi] = max(val_acc);
        f0_ta = tst_acc(bi);
        
        load(F1{f});
        
        f1_t1 = tst_acc(1 ); [~, bi] = max(val_acc);
        f1_ta = tst_acc(bi);
        
        df = [df; [f1_t1-f0_t1 f1_ta-f0_ta]];
    end
end

% IBP-LRC over only LRC
fprintf('%f\n', mean(df(:,2)*100));