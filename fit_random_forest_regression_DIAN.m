clear;clc;close all;
tbl = readtable('/Users/jiaxintu/Library/CloudStorage/Box-Box/NLA-DIAN/Brain_Revision/Data/FinalSubjectList_datecorrected_Location of errors FIXED CCS FC included for Revision 220918.xlsx');
tbl(isnan(tbl.Amyloid),:) = []; 
%% define which network pair
netFC =tbl.DMNCO;
netFCstr = 'DMN-CO';

% %% model 1
% disp('== model 1 ==')
% X = [tbl.Age, tbl.Education tbl.Sex == 1, netFC];
% Y = tbl.CognitiveCompositeScore;
% X_names = {'Age','Education','Sex',netFCstr};
% rng('default');
% ens = fitrensemble(X,Y,'Method','Bag','PredictorNames',X_names,'ResponseName','CCS');
% imp = predictorImportance(ens);
% disp(['Variables:',X_names])
% disp(['importance:',num2cell(imp)])
% 
% Y_hat = resubPredict(ens);
% RSS = sum((Y(ens.RowsUsed)-Y_hat).^2);
% k = size(ens.X,2); n = size(ens.X,1);
% delta_AIC = 2*k + n*log(RSS); % reference: wikipedia, https://en.wikipedia.org/wiki/Akaike_information_criterion
% fprintf('delta_AIC = %2.4f\n\n',delta_AIC)
% %% model 2
% disp('== model 2 ==')
% X = [tbl.Age, tbl.Education tbl.Sex == 1, netFC,tbl.logNfL];
% Y = tbl.CognitiveCompositeScore;
% X_names = {'Age','Education','Sex',netFCstr,'logNfL'};
% rng('default');
% ens = fitrensemble(X,Y,'Method','Bag','PredictorNames',X_names,'ResponseName','CCS');
% imp = predictorImportance(ens);
% disp(['Variables:',X_names])
% disp(['importance:',num2cell(imp)])
% 
% Y_hat = resubPredict(ens);
% RSS = sum((Y(ens.RowsUsed)-Y_hat).^2);
% k = size(ens.X,2); n = size(ens.X,1);
% delta_AIC = 2*k + n*log(RSS); 
% fprintf('delta_AIC = %2.4f\n\n',delta_AIC);
%% model 3
disp('== model 3 ==')
X = [tbl.Age, tbl.Education tbl.Sex == 1, netFC,tbl.logNfL,tbl.Amyloid];
Y = tbl.CognitiveCompositeScore;
X_names = {'Age','Education','Sex',netFCstr,'logNfL','Amyloid'};

rng('default');
ens = fitrensemble(X,Y,'Method','Bag','PredictorNames',X_names,'ResponseName','CCS');

% see that model converges
oobLall = oobLoss(ens,'Mode','cumulative');
% resubLall = resubLoss(ens,'Mode','cumulative');
figure;hold on;
plot(oobLall,'LineWidth',2);
% plot(resubLall);
xlabel('Number of trees');
ylabel('out-of-bag MSE');
set(gca,'FontSize',15);
title(netFCstr);
print(['RandomForestModelConvergence',netFCstr],'-dpdf');

%%
imp = predictorImportance(ens);
disp(['Variables:',X_names])
disp(['importance:',num2cell(imp)])

Y_hat = resubPredict(ens);
RSS = sum((Y(ens.RowsUsed)-Y_hat).^2);
TSS =  sum((Y(ens.RowsUsed)-mean(Y(ens.RowsUsed))).^2);
Rsquared = 1-RSS/TSS;
fprintf('Rsquared = %2.2f\n',Rsquared)

k = size(ens.X,2); n = size(ens.X,1);
delta_AIC = 2*k + n*log(RSS);
fprintf('delta_AIC = %2.4f\n',delta_AIC)

resubL = resubLoss(ens);
oobL = oobLoss(ens);

fprintf('MSE(resubstitution) = %2.2f\n',resubL)
fprintf('MSE(out-of-bag) = %2.2f\n\n',oobL)
%%
figure;
barh(imp);
yticklabels(X_names);
xlabel('Variable Importance');
title(netFCstr);
set(gca,'FontSize',15);
print(['RandomForestVariableImportance',netFCstr],'-dpdf');


% screenshot the command window