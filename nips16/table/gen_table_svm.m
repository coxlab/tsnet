M = {'SS', 'TS'};
D = {'mnist' 'cifar10' 'svhn2'};

for d = 1:length(D)
    
    F = sort(strsplit(ls(['svm/*/' D{d} '*.mat'])));
    
    for m = 1:length(M)
        
        v = 0;
        l = 0;
        t = 0;
        n = '';
        s = 0; sA = [];
        
        for f = 1:length(F)
            
            if isempty(F{f}), continue; end
            
            load(F{f});
            %plot([Rss(:,1) Rts(:,1)]); pause;
            
            if m == 1
                val_acc = Rss(2:end,1);
                tst_acc = Rss(2:end,2);
                time    = Rss(2:end,3);
            else
                val_acc = Rts(2:end,1);
                tst_acc = Rts(2:end,2);
                time    = Rts(2:end,3);
            end
            
            [bv, bi] = max(val_acc);
            
            if bv > v
                v = bv;
                l = bi;
                t = tst_acc(bi);
                n = F{f};
                s = time(bi);
            end
            
            sA = [sA time'];
        end
        
        if m == 1
            sAn = sA;
            sn  = s;
        end
        
        fprintf('%s mode %s best model: %s %d %5.2f %5.1f %5.1f\n', D{d}, M{m}, n, l, 100-t, s/sn, mean(sA./sAn));
    end
    
    fprintf([repmat('-',[1 80]) '\n']);
end