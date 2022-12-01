% test MatCont ODE

%global x v s h f xlc vlc slc hlc flc opt

opt=contset;opt=contset(opt,'Singularities',1);
[x0,v0]=init_EP_EP(@test_ODE,[0;0;0;0;0;0],-1,[1]);

[x,v,s,h,f]=cont(@equilibrium,x0,[],opt);

x1=x(1:6,s(2).index);p=x(end,s(2).index);

[x0,v0]=init_H_LC(@test_ODE,x1,p,[1],1e-4,20,4);

opt = contset(opt,'MaxNumPoints',200);
opt = contset(opt,'Multipliers',1);
opt = contset(opt,'Adapt',1);

[xlc,vlc,slc,hlc,flc]=cont(@limitcycle,x0,v0,opt);

figure
axes
plotcycle(xlc,vlc,slc,[size(xlc,1) 1 2]);