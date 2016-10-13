D = {'mnist' 'cifar10' 'svhn2'};
M = {'0' '1'};

for d = D
    for m = M

        v1 = 0;  va = 0;
        t1 = 0;  ta = 0;
        n1 = ''; na = '';
        s1 = 0;  sa = 0;
        s = [];
        
        F = strsplit(ls(['cnn/' d{1} '-' m{1} '*.mat']));
        
        for f = F
            
            if isempty(f{1}), continue; end
            
            load(f{1});
            
            if val_acc(1) > v1
                v1 = val_acc(1);
                t1 = tst_acc(1);
                n1 = f{1};
                s1 = mean(time);
            end
            
            [bv, bi] = max(val_acc);
            if bv > va
                va = bv;
                ta = tst_acc(bi);
                na = f{1};
                sa = mean(time);
            end
            
            s = [s time];
        end
        
        s = mean(s);
        
        if m{1} == '0'
            sn  = s;
            s1n = s1;
            san = sa;
        end
        
        fprintf('%s m-%s e1 %s %.2f %.1f     \n', d{1}, m{1}, n1, (1-t1)*100, s1/s1n);
        fprintf('%s m-%s ef %s %.2f %.1f %.1f\n', d{1}, m{1}, na, (1-ta)*100, sa/san, s/sn);
    end
end
