function Rlie = gEM(a,b,sigma,t,tInd,dW)
%%GEM geometric Euler-Maruyama
% x - coefficients for linear combination
% output: M - solution of SDE in Lie group
d = 4;
N = length(t);
M = size(dW,2);
dt = t(end)./(N+1);

% Li=zeros((d-1)^2,N,M);
dW = reshape(dW,1,N-1,M);
W=zeros(1,N,M);
W(1,2:end,:)=dW;
W=cumsum(W,2);
%Euler-Maruyama-increments
Li=b.*t+sigma.*W;
Li=cumsum(abs(Li).^a,2).*dt;
% fix numerical error
% Li(Li<0)=0;
L = generatorMatConst(Li);
dL=diff(L,1,3);
dL=reshape(dL,d,d,(N-1)*M);

dLexp = zeros(size(dL));

parfor twi=1:size(dLexp,3)
    dLexp(:,:,twi)=expm(dL(:,:,twi));
end

dLexp=reshape(dLexp,d,d,N-1,M);
Rlie=zeros(d,d,length(tInd),M);
temp=repmat(eye(d),1,1,1,M);

k=1;
if tInd(k)==1
    Rlie(:,:,k,:)=temp;
    k=k+1;
end
for ti=1:1:N-1
    temp=pagemtimes(squeeze(temp),squeeze(dLexp(:,:,ti,:)));
    if tInd(k)==ti+1
        Rlie(:,:,k,:)=reshape(temp,d,d,1,M);
        k=k+1;
    end
end

end