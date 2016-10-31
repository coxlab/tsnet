cm = lines(3);

for d = 1:3
    
    load(sprintf('cnn/cifar10-0-%d-%d-%d-%d.mat', repmat(d, [1 4])));
    
    E = length(acc);
    t = mean(time);
    
    loglog((1:E)*t, (1-val_acc)*100, ':', 'LineWidth', 1.5, 'Color', cm(d,:)); hold on;
end

for d = 1:3
    load(sprintf('cnn/cifar10-1-%d-%d-%d-%d.mat', repmat(d, [1 4])));
    
    E = length(acc);
    t = mean(time);
    
    loglog((1:E)*t, (1-val_acc)*100, '-', 'LineWidth', 1.5, 'Color', cm(d,:)); hold on;
end

axis tight;
set(gca,'YTick',30:10:80);
set(gca,'YTickLabel',strsplit(num2str(30:10:80,'%3d%%')));
set(gca,'XTick',[1e1 1e2 1e3]);
set(gca,'XTickLabel',{'10' '100' '1000'});

l = [];
l = [l {'SS {\it{L}}=3+1' 'SS {\it{L}}=6+2' 'SS {\it{L}}=9+3'}];
l = [l {'TS {\it{L}}=3+1' 'TS {\it{L}}=6+2' 'TS {\it{L}}=9+3'}];

h2 = legend(l);
xlabel('Seconds');
ylabel('Error Rate');