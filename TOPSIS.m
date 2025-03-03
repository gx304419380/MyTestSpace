clc;
clear all;
% Step 1: ��������
% ������m��������n������ָ��
X = xlsread('1', 'D3:G11'); % ������߾��󣬴�СΪ m �� n
X = transpose(X);
[m,n]=size(X);
weight = transpose(xlsread('1','H3:H11'));

% Step 2: ָ������Ȩ��׼��
% ������ֵ����Ϊ�ţ����б�׼��
for i=1:n
    normalized_X(:,i) = (X(:,i)- min(X(:,i)))./( max(X(:,i))- min(X(:,i))); % ָ���׼����ÿ��ָ����������ֵ
end

% ������ֵС��Ϊ�ţ����б�׼��
%normalized_X = (max(X(:))- X)./( max(X(:))- min(X(:))); % ָ���׼����ÿ��ָ����������ֵ
% % ����ÿ��ָ�����ֵ
% k=1./log(m);
% p = normalized_X ./ sum(normalized_X); % ����ÿ��������ÿ��ָ���ϵ����Ȩ��
% result = log(p);
% result(p == 0) = 0;
% h = -k.*sum(p .* result); % ����ÿ��ָ�����Ϣ��ֵ
% g=1-h;% ����ÿ��ָ��ı���ϵ��
% weight = g ./ sum(g,2); % ����ÿ��ָ���Ȩ��
% % ��ʾ����õ���Ȩ��
% disp('Ȩ�أ�');
% disp(weight);

Z=weight.*normalized_X;
% disp('��Ȩ��׼������');
% disp(Z);

%Step 3:����ŷ�Ͼ������ɫ������
positive_Z = max(Z); % ������⣬ÿ��ָ������ֵ
negative_Z = min(Z); % ������⣬ÿ��ָ�����Сֵ
distance_positive = sqrt(sum((Z- positive_Z) .^ 2, 2)); % ��������ŷ�Ͼ���
distance_negative = sqrt(sum((Z- negative_Z) .^ 2, 2)); % ��������ŷ�Ͼ���
abs_positive=abs(positive_Z - Z);
abs_negative=abs(negative_Z - Z);
R=0.5;%�ֱ�ϵ����
a_positive =(min(abs_positive(:))+R*max(abs_positive(:)))./(abs_positive+R*max(abs_positive(:)));% ����ɫ����ϵ������
a_negative =(min(abs_negative(:))+R*max(abs_negative(:)))./(abs_negative+R*max(abs_negative(:)));% ����ɫ����ϵ������
r_positive=sum(a_positive,2)./n;%����ɫ������
r_negative=sum(a_negative,2)./n;%����ɫ������

%Step 4:��������������Ⱥ���������
%�����ٴ���
R_positive=r_positive./max(r_positive(:));
R_negative=r_negative./max(r_negative(:));
D_positive=distance_positive./max(distance_positive(:));
D_negative=distance_negative./max(distance_negative(:));
%����ƫ�ó̶�
a1=0.5;
a2=1-a1;
T_positive=a1*D_negative+a2*R_positive;
T_negative=a1*D_positive+a2*R_negative;
%�������������
S_positive=T_positive./(T_positive+T_negative);
[S_positive_2,index] = sort(S_positive,'descend');
%�����S_positive_2������õ�������index������S_positive_2�ж�S_positive��������
disp('������������Ƚ�����������У���');
disp(index);