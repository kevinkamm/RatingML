function [t,W,R,varargout]=loadData(relDir,AE,G,lenSeq,batch,epochs,N,K,months,varargin)
path=[pwd,'/',relDir,'/',...
      sprintf('AE%d_G%d_lenSeq%d_batch%d_epochs%d_N%d',...
              AE,G,lenSeq,batch,epochs,N)];
saveMat = true;
loadMat = true;
for iVar = 1:1:length(varargin)
    switch varargin{iVar}
        case 'save'
            saveMat = varargin{iVar+1};
        case 'load'
            loadMat = varargin{iVar+1};
    end
end
loaded = true;
if loadMat
    if exist([path,'/','R.mat'])
        R=load([path,'/','R.mat']);
        R=R.R;
    else
        loaded=false;
    end
    if exist([path,'/','W.mat'])
        W=load([path,'/','W.mat']);
        W=W.W;
    else
        loaded=false;
    end
    if exist([path,'/','t.mat'])
        t=load([path,'/','t.mat']);
        t=t.t;
    else
        loaded=false;
    end
end
if ~loaded
    files = dir([path,'/*.csv']);
    M=length(files)/(lenSeq+1);
    R=zeros(K,K,lenSeq,M);
    W=zeros(N,M);
    t=zeros(N,1);
    monthMap=zeros(max(months),1);
    for i=length(months):-1:1
        monthMap(1:months(i))=i;
    end
    warning off
    for wi =1:length(files)
        file=[files(wi).folder,'/',files(wi).name];
        temp=readtable(file);
        if contains(file,'W')
            currW = extractBetween(files(wi).name,wildcardPattern+'_','.'+wildcardPattern);
            currW = str2num(currW{1})+1;
            if t(end)==0
                t=temp.Time;
            end
            W(:,currW)=temp.BrownianPath;
        elseif contains(file,'R')
            currW = extractBetween(files(wi).name,wildcardPattern+'_','_'+wildcardPattern);
            currW = str2num(currW{1})+1;
            currMonth = extract(file,digitsPattern+'.csv');
            currMonth = extract(currMonth{1},digitsPattern);
            currMonth = str2num(currMonth{1});
            R(:,:,monthMap(currMonth),currW)=temp{1:end,2:end};
        end
    end
    warning on
end
if saveMat && ~loaded
    save([path,'/','R.mat'],'R')
    save([path,'/','W.mat'],'W')
    save([path,'/','t.mat'],'t')
end
end