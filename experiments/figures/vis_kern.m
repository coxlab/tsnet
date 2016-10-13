clear; close all;

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
l = [l {'SS {\it{l}}=1' 'SS {\it{l}}=2' 'SS {\it{l}}=3'}];
l = [l {'TS {\it{l}}=1' 'TS {\it{l}}=2' 'TS {\it{l}}=3'}];

legend(l,'location','southwest');
set(gcf, 'Position', [0 0 400 300], 'PaperPositionMode', 'auto');

%print(gcf, '-dpdf', 'vis_kern.pdf');
%gs -dSAFER -dNOPLATFONTS -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sPAPERSIZE=letter -dCompatibilityLevel=1.4 -dPDFSETTINGS=/printer -dCompatibilityLevel=1.4 -dMaxSubsetPct=100 -dSubsetFonts=true -dEmbedAllFonts=true -sOutputFile=e_vis_kern.pdf -f vis_kern.pdf
