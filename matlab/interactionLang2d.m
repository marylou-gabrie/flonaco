%% Integrator for the Langevin equation
%
%  dx_i = (1/N) \sum_{j=1}^N \nabla W(x_i-x_j) dt + \sqrt{2\beta^{-1}} dW
%
%% Set-up

% parameters
kT = 0.095;            % diffusion coeff
a = 4;             % attraction range
b = 1;              % intensity

% grid
Nx = 2^9;
x = linspace(0,1,Nx+1);
x = x(1:Nx);
y = x;
[xx,yy] = ndgrid(x,y);

% attracting potential
W = @(x,y) -b*exp(a*cos(2*pi*x)+a*cos(2*pi*y)-2*a);
dWx = @(x,y) -a*2*pi*sin(2*pi*x).*W(x,y);
dWy = @(x,y) -a*2*pi*sin(2*pi*y).*W(x,y);
W1 =  W(xx,yy);        
Wk = fft2(W1);


%% Lang
n1 = 200;
Nt = 1e4;
dt = 1e-4;
xt = zeros(n1,Nt);
X = 0.5+0*rand(n1,1);
Y = 0.5+0*rand(n1,1);
xt(:,1) = sort(X);
xt0 = zeros(n1,Nt);
X0 = rand(n1,1);
Y0 = rand(n1,1);
xt0(:,1) = sort(X0);

E1 = zeros(1,Nt);
E1(1) = .5*sum(mean(W(repmat(X,1,n1)-repmat(X',n1,1),repmat(Y,1,n1)-repmat(Y',n1,1)),2));

E0 = zeros(1,Nt);
E0(1) = .5*sum(mean(W(repmat(X0,1,n1)-repmat(X0',n1,1),repmat(Y0,1,n1)-repmat(Y0',n1,1)),2));

%%
tic
for j=2:Nt
    dX = -mean(dWx(repmat(X,1,n1)-repmat(X',n1,1),repmat(Y,1,n1)-repmat(Y',n1,1)),2);
    dY = -mean(dWy(repmat(X,1,n1)-repmat(X',n1,1),repmat(Y,1,n1)-repmat(Y',n1,1)),2);
    X = mod(X + dX*dt + sqrt(2*kT*dt)*randn(n1,1),1);
    Y = mod(Y + dY*dt + sqrt(2*kT*dt)*randn(n1,1),1);
    
    E1(j) = .5*sum(mean(W(repmat(X,1,n1)-repmat(X',n1,1),repmat(Y,1,n1)-repmat(Y',n1,1)),2));
    xt(:,j) = sort(X);
    
    dX0 = -mean(dWx(repmat(X0,1,n1)-repmat(X0',n1,1),repmat(Y0,1,n1)-repmat(Y0',n1,1)),2);
    dY0 = -mean(dWy(repmat(X0,1,n1)-repmat(X0',n1,1),repmat(Y0,1,n1)-repmat(Y0',n1,1)),2);
    X0 = mod(X0 + dX0*dt + sqrt(2*kT*dt)*randn(n1,1),1);
    Y0 = mod(Y0 + dY0*dt + sqrt(2*kT*dt)*randn(n1,1),1);
    
    E0(j) = .5*sum(mean(W(repmat(X0,1,n1)-repmat(X0',n1,1),repmat(Y0,1,n1)-repmat(Y0',n1,1)),2));
    xt0(:,j) = sort(X0);
end
toc

%%
scr_siz = get(0,'ScreenSize') ;
fig_size = floor([.1*scr_siz(3) .1*scr_siz(4) .9*scr_siz(3) .6*scr_siz(4)]);

fig4=figure(20);clf;
subplot(3,1,1)
plot((0:Nt-1)*dt,E1/n1,'LineWidth',2);
hold on
plot((0:Nt-1)*dt,E0/n1,'LineWidth',2);
legend('traj. 1','traj. 2')
xlabel('time','FontSize',16,'FontAngle','italic');
ylabel('energy','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',24);
subplot(3,1,2)
plot((0:Nt-1)*dt,xt');
ylim([0 1])
xlabel('time','FontSize',16,'FontAngle','italic');
ylabel('Lang. traj. 1','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',24);
subplot(3,1,3)
plot((0:Nt-1)*dt,xt0');
ylim([0 1])
xlabel('time','FontSize',16,'FontAngle','italic');
ylabel('Lang. traj. 2','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',24);
fig4.Position = fig_size;

nbinss = 6e1;
fig5 = figure(21);clf;
histogram(E1/n1,nbinss,'normalization','PDF');
hold on
histogram(E0/n1,nbinss,'normalization','PDF');
ylabel('density of states','FontSize',16,'FontAngle','italic');
set(gca,'FontSize',24);
set(gca, 'YScale', 'log')
fig5.Position = fig_size;

