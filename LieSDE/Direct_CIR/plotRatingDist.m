function Figures=plotRatingDist(Rcal,Rgan,varargin)
backgroundColor='w';
textColor = 'k';
figureRatio = 'fullScreen';
visible='on';
for iV=1:2:length(varargin)
    switch varargin{iV}
        case 'backgroundColor'
            backgroundColor = varargin{iV+1};
        case 'figureRatio'
            figureRatio = varargin{iV+1};
        case 'textColor'
            textColor  = varargin{iV+1};
        case 'visible'
            visible  = varargin{iV+1};
    end
end
Figures={};
% dist='normal';
% dist='lognormal';
dist='beta';
nbins=20;
for ti = 1:1:size(Rcal,3)
    fig=newFigure('backgroundColor',backgroundColor,...
                  'textColor',textColor,...
                  'figureRatio',figureRatio,...
                  'visible',visible);
    tL=tiledlayout(size(Rcal,1)-1,size(Rcal,2));
    Figures{end+1}=fig;
    for ri = 1:1:size(Rcal,1)-1
        for rj = 1:1:size(Rcal,2)
            nexttile;
            hCal=histfit(reshape(Rcal(ri,rj,ti,:),[],1),nbins,dist);hold on;
            hCal(1).FaceColor=[0 0.4470 0.7410];
            hCal(1).FaceAlpha=.5;
            hCal(2).Color='b';
            hCal(2).LineStyle='--';
            hGan=histfit(reshape(Rgan(ri,rj,ti,:),[],1),nbins,dist);
            hGan(1).FaceColor=[0.6350 0.0780 0.1840];
            hGan(1).FaceAlpha=.5;
            hGan(2).Color=[0.8500 0.3250 0.0980];
            hGan(2).LineStyle='-';
        end
    end
    lgd = legend({'Rcal hist',['Rcal ', dist,' dist'],'Rgan hist',['Rgan ', dist,' dist']},'NumColumns',4);
    lgd.Layout.Tile = 'south';
end
end
% dark blue: [0 0.4470 0.7410]
% orange: [0.8500 0.3250 0.0980]
% yellow: [0.9290 0.6940 0.1250]
% purple: [0.4940 0.1840 0.5560]
% green: [0.4660 0.6740 0.1880]
% light blue: [0.3010 0.7450 0.9330]
% red: [0.6350 0.0780 0.1840]