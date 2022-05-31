function Rlie = gEM(a,b,sigma,t,tInd,dW)
%%GEM geometric Euler-Maruyama
% x - coefficients for linear combination
% output: M - solution of SDE in Lie group

d = 4;
N = length(t);
M = size(dW,2);
dt = t(end)./(N+1);

Li=zeros((d-1)^2,length(tInd),M);
dW = reshape(dW,1,N-1,M);

%Euler-Maruyama
k=1;
temp=zeros((d-1)^2,1,M);
if tInd(k)==1
    Li(:,k,:)=temp;
    k=k+1;
end
for ti=1:1:N-1
    temp=temp+a.*(b-temp).*dt + sigma.* sqrt(max(temp,0)).*dW(1,ti,:);
    if tInd(k)==ti+1
        Li(:,k,:)=temp;
        k=k+1;
    end
end

% fix numerical error
Li(Li<0)=0;
L = generatorMatConst(Li);
L=reshape(L,d,d,(length(tInd))*M);

Lexp = zeros(size(L));

parfor twi=1:size(L,3)
    Lexp(:,:,twi)=expm(L(:,:,twi));
end

Rlie=reshape(Lexp,d,d,length(tInd),M);

end