function y=objectiveLsqnonlin(x,t,tInd,dW,MuGAN,SigmaGAN,SkewGAN,KurtGAN)
a=x(1:9);
b=x(10:18);
sigma=x(19:27);
Rlie = gEM(a,b,sigma,t,tInd,dW);

[MuLie,SigmaLie,SkewLie,KurtLie]=ratingMoments(Rlie);
y1=abs(MuLie-MuGAN);
y2=10.*abs(SigmaLie-SigmaGAN);
y3=1.*abs(SkewLie-SkewGAN);
y4=1.*abs(KurtLie-KurtGAN);
y=[y1(:);y2(:);y3(:);y4(:)];

end