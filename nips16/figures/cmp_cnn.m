clear; close all;

%set(0, 'defaultAxesFontName','DejaVu Sans');
%set(0, 'defaultTextFontName','DejaVu Sans');

subplot(1,2,1); cmp_cnn_acc;
subplot(1,2,2); cmp_cnn_time;

set(gcf, 'Position', [0 0 1050 350], 'PaperPositionMode', 'auto', 'PaperOrientation', 'landscape');
h1.Position(1) = h1.Position(1) - 0.001;
h1.Position(2) = h1.Position(2) - 0.015;
h2.Position(1) = h2.Position(1) + 0.001;
h2.Position(2) = h2.Position(2) + 0.020;

%set(gcf, 'Color', 'none');
