clear all; close all; fclose('all');gpuDevice(1);rng(0);
pool=gcp('nocreate');
if isempty(pool)
%     pool=parpool('local');
    pool=parpool('threads');
end
%% Load data
% parameters of autoencoder
AE=560;
% parameters of generator
G=854;
% length of time-series
lenSeq=4;
% batch size
batch=128;
% epochs
epochs=40;
% number of time steps for Brownian motion
N=361;
% number of different ratings
K=4;
% months of time series
months=[1,3,6,12];
T = months(end)/12;

% relative path to data
relDir='Data';

ticLoadData=tic;
[~,~,Rgan]=loadData(relDir,AE,G,lenSeq,batch,epochs,N,K,months);
ctimeLoadData=toc(ticLoadData);
fprintf('Elapsed time for loading data %1.3f\n',ctimeLoadData);
% order of Lie Algebra basis
[~,basisOrder]=lieAlgebraBasis(K);
%% Figure settings
backgroundColor='w';
textColor='k';
%% Calibration
optimizer='lsqnonlin';
M=1000;
N=200*12+1;
ticCal=tic;
[params,errCal,dW,t,tInd]=calibrate(months,T,N,M,Rgan,'Optimizer',optimizer);
ctimeCal=toc(ticCal);
fprintf('Elapsed time for calibration %g s with error %1.3e\n',...
        ctimeCal,errCal);
aCal=params(1:9);
bCal=params(10:18);
sigmaCal=params(19:27);
paramTable = array2table([aCal,bCal,sigmaCal],...
                         'VariableNames',{'a','b','sigma'},...
                         'RowNames',basisOrder);
disp('Parameters after calibration')
disp(paramTable)
%% Simulation with calibrated parameters
ticSim=tic;
Rcal = gEM(aCal,bCal,sigmaCal,t,1:1:length(t),dW);
ctimeSim=toc(ticSim);
fprintf('Elapsed time for simulation %g s\n',...
        ctimeSim);
%%
figRD=plotRatingDist(Rcal(:,:,tInd,:),Rgan(:,:,:,1:M));
%%
figTra=plotTrajectories(t,tInd,Rcal,Rgan,...
                    'backgroundColor',backgroundColor,...
                    'textColor',textColor);
%%
[mDCgan,sDDgan,dMLgan,iRSgan,rSOgan]=ratingProperties(Rgan,months);
[mDCcal,sDDcal,dMLcal,iRScal,rSOcal]=ratingProperties(Rcal(:,:,tInd,:),months)
%% Output
fileName = sprintf('CIR_%s_N%d_M%d',optimizer,N,M);
compileLatex=true;
output(fileName,...
       N,M,...
       ctimeCal,...
       errCal,...
       paramTable,...
       mDCgan,sDDgan,dMLgan,iRSgan,rSOgan,...
       mDCcal,sDDcal,dMLcal,iRScal,rSOcal,...
       figRD,figTra,...
       'backgroundColor',backgroundColor,...
       'textColor',textColor,...
       'compileLatex',compileLatex)
disp('Done')