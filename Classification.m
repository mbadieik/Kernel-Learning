% Learning kernels with Random Features, example script

clear all;
close all;
load adult.mat
clc;
rng(7117);
kk=0;
Xtrain=x_t(1:35,:);
Xtest=x_test(1:35,:);
DD=size(Xtrain);
d=DD(1);
ytrain=y_t';
ytest=z_test';
BB=20:20:1000; 
for KKK=1:length(BB)
    KKK
    Nw=BB(KKK) %Number of Random Features
kk=kk+1;
for Trial=1:10
    Trial


rho = Nw*0.005;
tol = 1e-11;
[Wopt, bopt, alpha, alpha_distrib,t0] = optimizeGaussianKernel(Xtrain, ytrain, Nw, rho, tol);
[Wopt_1,bopt_1,t1] = MeanFieldOptimization(Xtrain,ytrain,d,Nw);
[Wopt_2,bopt_2,t2] = RadialKernel1(Xtrain,ytrain,d,Nw);
Time_IS(Trial,KKK)=t0;
Time_SGD(Trial,KKK)=t1;
Time_Alg(Trial,KKK)=t2;


%%
D = Nw;
% % generate parameters for the optimized kernel
[D_opt, W_opt, b_opt] = createOptimizedGaussianKernelParams(D, Wopt, bopt, alpha_distrib);
% % create optimized features using the training data and test data
Z_opt_train = createRandomFourierFeatures(D, W_opt, b_opt, Xtrain);
Z_opt_test = createRandomFourierFeatures(D, W_opt, b_opt, Xtest);
% 
% %Z_opt_train_subset = createRandomFourierFeatures(D, Wopt_1_subset, bopt_1_subset, Xtrain);
% %Z_opt_test_subset = createRandomFourierFeatures(D, Wopt_1_subset, bopt_1_subset, Xtest);
% 
% 
% % Generate regular Gaussian features for comparison
W = randn(d,D);
b = rand(1,D)*2*pi;
Z_train = createRandomFourierFeatures(D, W, b, Xtrain);
Z_test = createRandomFourierFeatures(D, W, b, Xtest);
% 
W = randn(10,D);
b = rand(1,D)*2*pi;

 
Z_opt_1_train = createRandomFourierFeatures(D, Wopt_1, bopt_1, Xtrain);
Z_opt_1_test = createRandomFourierFeatures(D, Wopt_1, bopt_1, Xtest);

Z_opt_2_train = createRandomFourierFeatures(D, Wopt_2, bopt_2, Xtrain);
Z_opt_2_test = createRandomFourierFeatures(D, Wopt_2, bopt_2, Xtest);


%Z_opt_3_train = createRandomFourierFeatures(D, Wopt_3, bopt_3, Xtrain);
%Z_opt_3_test = createRandomFourierFeatures(D, Wopt_3, bopt_3, Xtest);

%%
% 
% %% Train models
% % For simplicity, train linear regression models (even though this is a
% % classification problem!)
% %meany = mean(ytrain);
% %meany_opt=meany;
% %meany_opt_1=meany;
% %lambda = .05;
% 
% %w_opt = (Z_opt_train * Z_opt_train' + lambda * eye(D_opt)) \ (Z_opt_train * (ytrain-meany));
% %w = (Z_train * Z_train' + lambda * eye(D)) \ (Z_train * (ytrain-meany));
% %w_opt_1 = (Z_opt_1_train * Z_opt_1_train' + lambda * eye(D)) \ (Z_opt_1_train * (ytrain-meany));
% 
% % Note that we don't bother scaling the features by sqrt(alpha) since we
% % can absorb that factor into w_opt for this ridge regression model
% 
% % If you have the ability to use smarter models, then you can try:
% % mdl = fitglm(Z_train', (ytrain+1)/2, 'Distribution', 'binomial');
% % or
mdl = fitcsvm(Z_train', ytrain, 'KernelFunction', 'linear', 'ClassNames', [-1, 1]);
w=mdl.Beta;
meany=mdl.Bias;

% 
mdl_opt=fitcsvm(Z_opt_train', ytrain, 'KernelFunction', 'linear', 'ClassNames', [-1, 1])
w_opt=mdl_opt.Beta;
meany_opt=mdl_opt.Bias;
% 
mdl_opt_1 = fitcsvm(Z_opt_1_train', ytrain, 'KernelFunction', 'linear', 'ClassNames', [-1, 1]);
w_opt_1=mdl_opt_1.Beta;
meany_opt_1=mdl_opt_1.Bias;

 
 mdl_opt_2 = fitcsvm(Z_opt_2_train', ytrain, 'KernelFunction', 'linear', 'ClassNames', [-1, 1]);
 w_opt_2=mdl_opt_2.Beta;
 meany_opt_2=mdl_opt_2.Bias;

% and then change the error computation code accordingly for the logistic
% regression or SVM models respectively.

%% errors
%calculate errors on training set
%disp(['Fraction of positives (train): ' num2str(sum(ytrain==1)/length(ytrain))])
%disp(' ')
[err_1(Trial,kk),fp_1(Trial,kk), fn_1(Trial,kk)] = computeError(Z_train, w, meany, ytrain);


[err_2(Trial,kk),fp_2(Trial,kk), fn_2(Trial,kk)] = computeError(Z_opt_train, w_opt, meany_opt, ytrain);

[err_3(Trial,kk),fp_3(Trial,kk), fn_3(Trial,kk)] = computeError(Z_opt_1_train, w_opt_1, meany_opt_1, ytrain);

[err_4(Trial,kk),fp_4(Trial,kk), fn_4(Trial,kk)] = computeError(Z_opt_2_train, w_opt_2, meany_opt_2, ytrain);

%disp(['Fraction of positives (test): ' num2str(sum(ytest==1)/length(ytest))])
[err_5(Trial,kk),fp_5(Trial,kk), fn_5(Trial,kk)] = computeError(Z_test, w, meany, ytest);

[err_6(Trial,kk),fp_6(Trial,kk), fn_6(Trial,kk)] = computeError(Z_opt_test, w_opt, meany_opt, ytest);

[err_7(Trial,kk),fp_7(Trial,kk), fn_7(Trial,kk)] = computeError(Z_opt_1_test, w_opt_1, meany_opt_1, ytest);

[err_8(Trial,kk),fp_8(Trial,kk), fn_8(Trial,kk)] = computeError(Z_opt_2_test, w_opt_2, meany_opt_2, ytest);

end

end


Nw = BB;

figure(10)
set(gca,'FontSize',18)
plot_distribution_prctile(Nw,Time_SGD(:,1:31),'Color',[0 0 0.8],'Prctile',[20 30 50]), hold on
plot_distribution_prctile(Nw,Time_IS(:,1:31),'Color',[.4 .5 0],'Prctile',[20 30 50]), hold on
plot_distribution_prctile(Nw,Time_Alg(:,1:31),'Color',[.1 .1 .1],'Prctile',[20 30 50]), hold on
A10=plot(Nw,mean(Time_SGD(:,1:31)),'LineWidth',4,'Color',[0 0 0.8],'LineStyle','-'), hold on
A11=plot(Nw,mean(Time_IS(:,1:31)),'LineWidth',4,'Color',[0.4 0.5 0],'LineStyle','-'), 
A22=plot(Nw,mean(Time_Alg(:,1:31)),'LineWidth',4,'Color',[0.1 0.1 0.1],'LineStyle','-'), 


legend([A10 A11 A22],'SGD','IS','Alg.1','Interpreter','latex')
xlabel('Number of Random Feature Samples ($N$)','Interpreter','latex','FontSize', 18) 
ylabel('Kernel Training Run-Time (Sec)','Interpreter','latex','FontSize', 18) 


figure(1) 
set(gca,'FontSize',18)
plot_distribution_prctile(Nw,err_1(:,1:31),'Color',[1 0 0],'Prctile',[20 30 50]), hold on
A1=plot(Nw,mean(err_1(:,1:31)),'LineWidth',2,'Color',[1 0 0],'LineStyle','--'), hold on

plot_distribution_prctile(Nw,err_2(:,1:31),'Color',[.4 .5 0],'Prctile',[20 30 50]), hold on
A2=plot(Nw,mean(err_2(:,1:31)),'LineWidth',2,'Color',[.4 .5 0],'LineStyle','--'), hold on

plot_distribution_prctile(Nw,err_3(:,1:31),'Color',[0 0 0.8],'Prctile',[20 30 50]), hold on
A3=plot(Nw,mean(err_3(:,1:31)),'LineWidth',2,'Color',[0 0 0.8],'LineStyle','--'), hold on

plot_distribution_prctile(Nw,err_4(:,1:31),'Color',[0.1 0.1 0.1],'Prctile',[20 30 50]), hold on
A4=plot(Nw,mean(err_4(:,1:31)),'LineWidth',2,'Color',[0.1 .1 .1],'LineStyle','--'), hold on



set(gca,'FontSize',18)
plot_distribution_prctile(Nw,err_5(:,1:31),'Color',[.8 0 0],'Prctile',[20 30 50]), hold on
A5=plot(Nw,mean(err_5(:,1:31)),'LineWidth',2,'Color',[.8 0 0],'LineStyle','-'), hold on

plot_distribution_prctile(Nw,err_6(:,1:31),'Color',[.4 .5 0],'Prctile',[20 30 50]), hold on
A6=plot(Nw,mean(err_6(:,1:31)),'LineWidth',2,'Color',[.4 .5 0],'LineStyle','-'), hold on

plot_distribution_prctile(Nw,err_7(:,1:31),'Color',[0 0 0.8],'Prctile',[20 30 50]), hold on
A7=plot(Nw,mean(err_7(:,1:31)),'LineWidth',2,'Color',[0 0 0.8],'LineStyle','-'), hold on

plot_distribution_prctile(Nw,err_8(:,1:31),'Color',[0.1 0.1 0.1],'Prctile',[20 30 50]), hold on
A8=plot(Nw,mean(err_8(:,1:31)),'LineWidth',2,'Color',[0.1 0.1 0.1],'LineStyle','-'), hold on

legend([A1 A2 A3 A4],'Gaussian','IS','SGD','Alg.1','Alg.1 $R=100$','Interpreter','latex')
xlabel('Number of Random Feature Samples (N)','Interpreter','latex','FontSize', 18) 
ylabel('Training/Test Error','Interpreter','latex','FontSize', 18) 
 
