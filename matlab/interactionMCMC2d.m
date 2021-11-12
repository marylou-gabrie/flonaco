%% Adaptive MCMC augmented by resampling for the Langevin equation
%
%  dx_i = (1/N) \sum_{j=1}^N \nabla W(x_i-x_j) dt + \sqrt{2\beta^{-1}} dW
%
%% Set-up

% parameters
kT = 0.08;            % diffusion coeff
a = 4;             % attraction range
b = 1;              % intensity

met1 = 1;       % 0 = rejection, 1 = conditional prob

% grid
Nx = 2^9;
xa = linspace(0,1,Nx+1);
[xxa,yya] = ndgrid(xa,xa);
x = xa(1:Nx);
[xx,yy] = ndgrid(x,x);

% attracting potential
W = @(x,y) -b*exp(a*cos(2*pi*x)+a*cos(2*pi*y)-2*a);
dWx = @(x,y) -a*2*pi*sin(2*pi*x).*W(x,y);
dWy = @(x,y) -a*2*pi*sin(2*pi*y).*W(x,y);
W1 =  W(xx,yy);        
Wk = fft2(W1);

scr_siz = get(0,'ScreenSize') ;
fig_size = floor([.1*scr_siz(3) .1*scr_siz(4) .9*scr_siz(3) .6*scr_siz(4)]);

%% Calculation of the MF density

u1 = exp(b*exp(a*cos(2*pi*(xx-.5))+a*cos(2*pi*(yy-.5))-2*a)/kT);
u1  = u1./sum(u1(:))*Nx^2;
u1k = fft2(u1);
ui = u1;
uik = fft2(ui);

tic
tol = 1e-9;
for i = 1:1e4 
    U2k = Wk.*uik/Nx^2;
    ui00 = ui;
    ui = exp(-ifft2(U2k,'symmetric')/kT);
    ui = ui/sum(ui(:))*Nx^2;
    uik = fft2(ui);
    if sum(sum(abs(ui00-ui)))/Nx^2<tol; break; end
end
uimax = max(ui(:));

fig1=figure(1);clf
subplot(1,2,1)
s=surf(xx,yy,ui);
s.EdgeColor = 'none';
colorbar
zlim([0,uimax])
set(gca,'FontSize',24);
fig1.Position = fig_size;

fig2=figure(2);clf
subplot(1,2,1)
s=surf(xx,yy,ui);
s.EdgeColor = 'none';
view(2)
xlim([0,1])
ylim([0,1])
axis square
colorbar
xlabel('x');
ylabel('y');
set(gca,'FontSize',24);
fig2.Position = fig_size;

%% Free energy
np = 1e2;
pp = linspace(0,1,np); Eih0 = zeros(size(pp)); uh = 1+0*ui; uhk = fft2(uh);
for ip = 1:np 
    uih = (1-pp(ip))*uh+pp(ip)*ui;
    uihk = (1-pp(ip))*uhk+pp(ip)*uik;
    Eih0(ip) = sum(sum(uih.*ifft2(Wk.*uihk,'symmetric')))*.5/Nx^4+kT.*sum(sum(uih.*log(uih)))/Nx^2;
end

figure(8); clf;
plot(pp,Eih0)
hold on
xlabel('x');
ylabel('Interpolated energy');
set(gca,'FontSize',24);


%% MCMC
uia = [ui ui(:,1)];
uia = [uia; ui(1,:) ui(1,1)];
rhoMF = griddedInterpolant(xxa,yya,uia);
npart = 1e4;
if met1 == 1
    uiax = sum(uia,2)'/Nx;
    Uiax = cumsum(uiax)/Nx;
    Tix = griddedInterpolant(Uiax,xa);
    Uiaxy = cumsum(uia,2)./repmat(uiax',1,Nx+1)/Nx;
    Tixy = scatteredInterpolant(reshape(xxa,(Nx+1)^2,1),reshape(Uiaxy,(Nx+1)^2,1),reshape(yya,(Nx+1)^2,1));
    Tixy.Method = 'natural';
    X = rand(npart,1);
    X = Tix(X);
    Y = rand(npart,1);
    Y = Tixy(X,Y);
else  
    npart0 = 1e7;
    npart = 1e4;
    X = rand(npart0,1);
    Y = rand(npart0,1);
    tt1 = rhoMF(X,Y)>uimax*rand(npart0,1);
    X = X(tt1);
    Y = Y(tt1);
    X = X(1:npart);
    Y = Y(1:npart);
end

figure(1)
subplot(1,2,2)
histogram2(X,Y,'normalization','PDF','FaceColor','flat')
xlim([0,1])
ylim([0,1])
zlim([0,uimax])
colorbar
xlabel('x');
ylabel('y');
set(gca,'FontSize',24);

figure(2)
subplot(1,2,2)
histogram2(X,Y,'normalization','PDF','DisplayStyle','tile','ShowEmptyBins','on')
xlim([0,1])
ylim([0,1])
axis square
colorbar
xlabel('x');
ylabel('y');
set(gca,'FontSize',24);

%%
n1 = 200;
npart0 = 2e5;
Nt = 1e4;
dt = 1e-4;
ntL = 4;
t11 = ((1:Nt)-1)*ntL*dt;
p = 0.5; q = 1-p;
hp = 1e-3;
gamp = 80;
zt = zeros(1,Nt);
zt(1) = log(p/q);
vz = 0;
xt = zeros(n1,Nt);
if rand<p
    if met1 == 1
        X = rand(n1,1);
        X = Tix(X);
        Y = rand(n1,1);
        Y = Tixy(X,Y);
    else
        tt1 = 1;
        while sum(tt1)<n1
            X = rand(npart0,1);
            Y = rand(npart0,1);
            tt1 = rhoMF(X,Y)>uimax*rand(npart0,1);
        end
        X = X(tt1);
        Y = Y(tt1);
        X = X(1:n1);
        Y = Y(1:n1);
    end
else
    X = rand(n1,1);
    Y = rand(n1,1);
end
xt(:,1) = sort(X);

E1 = zeros(1,Nt);
E1(1) = .5*sum(mean(W(repmat(X,1,n1)-repmat(X',n1,1),repmat(Y,1,n1)-repmat(Y',n1,1)),2));

ent1 = zeros(1,Nt);
ent1(1) = max(sum(log(rhoMF(X,Y)))+log(p),log(q));

%%
acc = zeros(Nt,1);
for j=2:Nt
    if rand<p  
        if met1 == 1
            X0 = rand(n1,1);
            X0 = Tix(X0);
            Y0 = rand(n1,1);
            Y0 = Tixy(X0,Y0);
        else
            tt1 = 1;
            while sum(tt1)<n1
                X0 = rand(npart0,1);
                Y0 = rand(npart0,1);
                tt1 = rhoMF(X0,Y0)>uimax*rand(npart0,1);
            end
            X0 = X0(tt1);
            Y0 = Y0(tt1);
            X0 = X0(1:n1);
            Y0 = Y0(1:n1);
        end
    else
        X0 = rand(n1,1);
        Y0 = rand(n1,1);
    end
    E10 = .5*sum(mean(W(repmat(X0,1,n1)-repmat(X0',n1,1),repmat(Y0,1,n1)-repmat(Y0',n1,1)),2));
    ent10 = max(sum(log(rhoMF(X0,Y0)))+log(p),log(q));
    MH1 = exp(-E10/kT+E1(j-1)/kT-ent10+ent1(j-1));
    if MH1>rand 
        X = X0; Y = Y0; E1(j) = E10;  ent1(j) = ent10; acc(j) = 1;
    else 
        E1(j) = E1(j-1); ent1(j) = ent1(j-1);
    end
    rhsvz = (sum(log(rhoMF(X,Y)))+log(p)>log(q))/p-(sum(log(rhoMF(X,Y)))+log(p)<log(q))/q;
    if ntL>0
        for jL=1:ntL
            dX = -mean(dWx(repmat(X,1,n1)-repmat(X',n1,1),repmat(Y,1,n1)-repmat(Y',n1,1)),2);
            dY = -mean(dWy(repmat(X,1,n1)-repmat(X',n1,1),repmat(Y,1,n1)-repmat(Y',n1,1)),2);
            X = mod(X + dX*dt + sqrt(2*kT*dt)*randn(n1,1),1);
            Y = mod(Y + dY*dt + sqrt(2*kT*dt)*randn(n1,1),1);
            rhsvz = rhsvz+(sum(log(rhoMF(X,Y)))+log(p)>log(q))/p-(sum(log(rhoMF(X,Y)))+log(p)<log(q))/q;
        end
        E1(j) = .5*sum(mean(W(repmat(X,1,n1)-repmat(X',n1,1),repmat(Y,1,n1)-repmat(Y',n1,1)),2));
        for kk = 1:10
            X = mod(X-mean(X)+.5,1);
            Y = mod(Y-mean(Y)+.5,1);
        end
        ent1(j) = max(sum(log(rhoMF(X,Y)))+log(p),log(q));
        rhsvz = rhsvz/(ntL+1);
    end
    vz = exp(-gamp*hp)*vz+(1-exp(-gamp*hp))*rhsvz;
    zt(j) = zt(j-1) + hp*vz;
    p = exp(zt(j))/(1+exp(zt(j))); q = 1-p;
    xt(:,j) = sort(X);
end
toc

fprintf(['n = ',num2str(n1),' kT = ',num2str(kT),' acceptance ratio = ',num2str(mean(acc)),' p = ',num2str(p),'\n'])

%%

fig4=figure(4);clf;
subplot(3,1,1)
plot(E1/n1);
xlabel('time','FontSize',16,'FontAngle','italic');
ylabel('energy','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',24);
subplot(3,1,2)
plot(xt');
ylim([0 1])
xlabel('time','FontSize',16,'FontAngle','italic');
ylabel('resampled traj.','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',24);
subplot(3,1,3)
plot(ent1/n1);
xlabel('time','FontSize',16,'FontAngle','italic');
ylabel('entropy','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',24);
fig4.Position = fig_size;

nbinss = 6e1;
fig5 = figure(5);clf;
subplot(1,2,1)
histogram(E1/n1,nbinss,'normalization','PDF');
ylabel('density of states','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',24);
set(gca, 'YScale', 'log')
subplot(1,2,2)
histogram(ent1/n1,nbinss,'normalization','PDF');
ylabel('PDF of entropy','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',24);
set(gca, 'YScale', 'log')
fig5.Position = fig_size;

figure(7);clf;
plot(t11,exp(zt)./(exp(zt)+1),'LineWidth',2);
hold on
xlabel('time','FontSize',16,'FontAngle','italic');
ylabel('p','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',24);
