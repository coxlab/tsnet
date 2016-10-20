folder = 'cnn/';
D = {'mnist' 'cifar10' 'svhn2'};

ssqrt = @(x)sign(x).*sqrt(abs(x));

cm = lines(3);
plot([0 0 NaN -10 10],[-10 10 NaN 0 0 ],'k:'); hold on;

for d = 1:length(D)
    
    df = [];
    
    F0 = sort(strsplit(ls([folder D{d} '-0*.mat'])));
    F1 = sort(strsplit(ls([folder D{d} '-1*.mat'])));
    
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
    
    df = df * 100;
    df = ssqrt(df);
    
    h(d) = scatter(df(:,1), df(:,2), 6.0^2, cm(d,:), 'filled');
end

xlim([-3.5 10]);
ylim([-sqrt(1.5) sqrt(16)]);

set(gca,'YTick',-1:4);
set(gca,'YTickLabel',{'-1%' '0%' '1%' '4%' '9%' '16%'});

set(gca,'XTick',-2:2:10);
set(gca,'XTickLabel',{'-4%' '0%' '4%' '16%' '36%' '64%' '100%'});

xlabel('One-pass Error Rate \Delta');
ylabel('Asymptotic Error Rate \Delta');

h1 = legend([h(1) h(2) h(3)], upper(D), 'Orientation', 'horizontal', 'Location', 'best');