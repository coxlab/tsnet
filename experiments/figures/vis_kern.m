clear; close all;

%set(0, 'defaultAxesFontName','DejaVu Sans');
%set(0, 'defaultTextFontName','DejaVu Sans');

ssrelu = @(theta_ss)          (sin(theta_ss)+(pi-theta_ss).*cos(theta_ss))/pi;
tsrelu = @(theta_ss, theta_ts)(1-theta_ss/pi).*cos(theta_ts);

theta = linspace(0,1);

ln = {   cos(pi*theta          )};
ss = {ssrelu(pi*theta          )};
ts = {tsrelu(pi*theta, pi*theta)};

for i = 1:3
    ts = [ts {tsrelu(acos(ss{end}), acos(ts{end}))}];
    ss = [ss {ssrelu(acos(ss{end})               )}];
end
    
figure; cm = lines(3); lw = 1.5;

plot(theta, ln{1}, 'LineWidth', lw, 'Color', cm(3,:)*1.00); hold on;

plot(theta, ss{1}, 'LineWidth', lw, 'Color', cm(1,:)*1.00);
plot(theta, ss{2}, 'LineWidth', lw, 'Color', cm(1,:)*0.75);
plot(theta, ss{3}, 'LineWidth', lw, 'Color', cm(1,:)*0.50);

plot(theta, ts{1}, 'LineWidth', lw, 'Color', cm(2,:)*1.00);
plot(theta, ts{2}, 'LineWidth', lw, 'Color', cm(2,:)*0.75);
plot(theta, ts{3}, 'LineWidth', lw, 'Color', cm(2,:)*0.50);

xlabel('\theta'); set(gca,'XTick',[0 0.5 1]); set(gca,'XTickLabel',{'0','0.5\pi','\pi'})
ylabel('\it{k}');

l = {'Linear'};
l = [l {'SS {\it{L}}=1' 'SS {\it{L}}=2' 'SS {\it{L}}=3'}];
l = [l {'TS {\it{L}}=1' 'TS {\it{L}}=2' 'TS {\it{L}}=3'}];

h = legend(l,'location','southwest');

set(gcf, 'Position', [0 0 400 300], 'PaperPositionMode', 'auto');
h.Position(1) = h.Position(1) - 0.01;
h.Position(2) = h.Position(2) - 0.04;

%set(gcf, 'Color', 'none');
