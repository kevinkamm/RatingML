function [params,err,dW,t,tInd]=calibrate(months,T,N,M,Rgan,varargin)

optimizer='lsqnonlin';
for iV=1:1:length(varargin)
    switch varargin{iV}
        case 'optimizer'
            optimizer=varargin{iV+1};
    end
end

t=linspace(0,T,N);
dW = sqrt(t(end)/(N+1)).*randn(N-1,M);
tInd=zeros(size(months));
for i=1:1:length(tInd)
    tInd(i)=find(t>=months(i)/12,1,'first');
end
[MuGAN,SigmaGAN,SkewGAN,KurtGAN]=ratingMoments(Rgan(:,:,end,:));
switch optimizer
    case 'fmincon'
        options = optimoptions('fmincon',...
                               'Display','iter',...
                               'StepTolerance',1e-10,...
                               'MaxFunctionEvaluations',10000,...
                               'UseParallel',true);
        lb=zeros(18,1);
        ub=1.*ones(18,1);
        x0=(ub+lb)./2;
        [params,err]=fmincon(@(x)objectiveFmincon(x,t,tInd(end),dW,...
                             MuGAN,SigmaGAN,SkewGAN,KurtGAN),...
                             x0,[],[],[],[],...
                             lb,ub,...
                             [],options);

    case 'lsqnonlin'
        lb=1e-4.*ones(27,1);
        ub=2.*ones(27,1);
        x0=(ub+lb)./2;
        options = optimoptions('lsqnonlin',...
                               'Display','iter',...
                               'UseParallel',true);
       
        [params,err] = lsqnonlin(@(x)objectiveLsqnonlin(x,t,tInd(end),dW,...
                        MuGAN,SigmaGAN,SkewGAN,KurtGAN),x0,lb,ub,options);
    otherwise
        error('Unknown optimizer')
end
end