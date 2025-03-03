clc;
clear all;
% Step 1: 输入数据
% 假设有m个样本和n个评价指标
X = xlsread('1', 'D3:G11'); % 输入决策矩阵，大小为 m × n
X = transpose(X);
[m,n]=size(X);
weight = transpose(xlsread('1','H3:H11'));

% Step 2: 指标矩阵加权标准化
% 假设数值大者为优，进行标准化
for i=1:n
    normalized_X(:,i) = (X(:,i)- min(X(:,i)))./( max(X(:,i))- min(X(:,i))); % 指标标准化，每个指标除以其最大值
end

% 假设数值小者为优，进行标准化
%normalized_X = (max(X(:))- X)./( max(X(:))- min(X(:))); % 指标标准化，每个指标除以其最大值
% % 计算每个指标的熵值
% k=1./log(m);
% p = normalized_X ./ sum(normalized_X); % 计算每个样本在每个指标上的相对权重
% result = log(p);
% result(p == 0) = 0;
% h = -k.*sum(p .* result); % 计算每个指标的信息熵值
% g=1-h;% 计算每个指标的变异系数
% weight = g ./ sum(g,2); % 计算每个指标的权重
% % 显示计算得到的权重
% disp('权重：');
% disp(weight);

Z=weight.*normalized_X;
% disp('加权标准化矩阵：');
% disp(Z);

%Step 3:计算欧氏距离与灰色关联度
positive_Z = max(Z); % 正理想解，每个指标的最大值
negative_Z = min(Z); % 负理想解，每个指标的最小值
distance_positive = sqrt(sum((Z- positive_Z) .^ 2, 2)); % 正理想解的欧氏距离
distance_negative = sqrt(sum((Z- negative_Z) .^ 2, 2)); % 负理想解的欧氏距离
abs_positive=abs(positive_Z - Z);
abs_negative=abs(negative_Z - Z);
R=0.5;%分辨系数；
a_positive =(min(abs_positive(:))+R*max(abs_positive(:)))./(abs_positive+R*max(abs_positive(:)));% 正灰色关联系数矩阵
a_negative =(min(abs_negative(:))+R*max(abs_negative(:)))./(abs_negative+R*max(abs_negative(:)));% 负灰色关联系数矩阵
r_positive=sum(a_positive,2)./n;%正灰色关联度
r_negative=sum(a_negative,2)./n;%负灰色关联度

%Step 4:样本的相对贴近度和优劣排序
%无量纲处理
R_positive=r_positive./max(r_positive(:));
R_negative=r_negative./max(r_negative(:));
D_positive=distance_positive./max(distance_positive(:));
D_negative=distance_negative./max(distance_negative(:));
%设置偏好程度
a1=0.5;
a2=1-a1;
T_positive=a1*D_negative+a2*R_positive;
T_negative=a1*D_positive+a2*R_negative;
%计算相对贴近度
S_positive=T_positive./(T_positive+T_negative);
[S_positive_2,index] = sort(S_positive,'descend');
%排序后，S_positive_2是排序好的向量，index是向量S_positive_2中对S_positive的索引。
disp('样本相对贴近度结果（降序排列）：');
disp(index);