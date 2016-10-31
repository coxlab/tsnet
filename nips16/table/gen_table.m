function gen_table(exp)

if     exp == 'mlp', folder = 'mlp/*/'; M = {'0' '1' '2' '3'};
elseif exp == 'cnn', folder = 'cnn/';   M = {'0' '1'        };
end

D = {'mnist' 'cifar10' 'svhn2'};

for d = 1:length(D)
    for m = 1:length(M)
        
        v1 = 0;  va = 0;
        t1 = 0;  ta = 0;
        n1 = ''; na = '';
        s1 = 0;  sa = 0; sA = [];
        
        F = sort(strsplit(ls([folder D{d} '-' M{m} '*.mat'])));
        
        for f = 1:length(F)
            
            if isempty(F{f}), continue; end
            
            load(F{f});
            
            if val_acc(1) > v1
                v1 = val_acc(1);
                t1 = tst_acc(1);
                n1 = F{f};
                s1 = mean(time);
            end
            
            [bv, bi] = max(val_acc);
            if bv > va
                va = bv;
                ta = tst_acc(bi);
                na = F{f};
                sa = mean(time);
            end
            
            sA = [sA mean(time)];
        end
        
        if m == 1
            sAn = sA;
            s1n = s1;
            san = sa;
        elseif m == 4
            sAn = sAn(1:3:end);
        end
        
        fprintf('%s mode %s one-epoch best model: %s %5.2f %5.1f %5.1f\n', D{d}, M{m}, n1, (1-t1)*100, s1/s1n, mean(sA./sAn));
        fprintf('%s mode %s all-epoch best model: %s %5.2f %5.1f %5.1f\n', D{d}, M{m}, na, (1-ta)*100, sa/san, mean(sA./sAn));
    end
    
    fprintf([repmat('-',[1 80]) '\n']);
end