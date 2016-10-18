function gen_table(exp)

if     exp == 'mlp', folder = 'mlp/*/'; M = {'0' '1' '2' '3'};
elseif exp == 'cnn', folder = 'cnn/';   M = {'0' '1'        };
end

D = {'mnist' 'cifar10' 'svhn2'};

for d = D
    for m = M

        v1 = 0;  va = 0;
        t1 = 0;  ta = 0;
        n1 = ''; na = '';
        s1 = 0;  sa = 0; sA = [];
        
        F = strsplit(ls([folder d{1} '-' m{1} '*.mat']));
        
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
            
            sA = [sA time];
        end
        
        sA = mean(sA);
        
        if m{1} == '0'
            sAn = sA;
            s1n = s1;
            san = sa;
        end
        
        fprintf('%s mode %s one-epoch best model: %s %5.2f %5.1f %5.1f\n', d{1}, m{1}, n1, (1-t1)*100, s1/s1n, sA/sAn);
        fprintf('%s mode %s all-epoch best model: %s %5.2f %5.1f %5.1f\n', d{1}, m{1}, na, (1-ta)*100, sa/san, sA/sAn);
    end
    
    fprintf([repmat('-',[1 80]) '\n']);
end