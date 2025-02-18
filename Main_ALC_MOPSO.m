% ������Ŀ������Ŀ-�����Դ�������ʹ����Ż�ѡ���붯̬����
% ��25��׹��������26��׷�粢����ȫ�������ڷ���
% �Ӹ��ɡ�Դ�����ҵģʽ�����Ƕȣ������ط���
% ����PSO�㷨,���ܹ������������������
% 2025.01.22
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc

tic;
Font_size = 12;   %�����С
format short;
warning off
set(0, 'defaultAxesFontName', 'Monospaced');
hwait = waitbar(0,'�������룬��ȴ�>>>>>>>>');
%% �û�����
n_sence     = 1;      % ���泡��ѭ��������1Ϊ���η�����= N_sence=100Ϊ�ೡ������
season      = '��';   % �Ļ򶬼�������
year        = 27;     % 26��ֻ��475MW�����27���⹲925MW����27�꿪ʼ����ȫ�����������Ŵ�������
Alc_year    = 25;     % ����ʹ������

% ����ΪĬ��ֵ�����޸�
Rate_Load   = 1.5;    % �����������������ֵ
Rate_Power  = 1.0;    % ����������
Rate_limt   = [0.85; 1]; %����Դ������Լ����95%-100%
Storage_max = 300;  %����ܶ������MW
Storage_min = 100;  %��С���ܶ������MW
Max_Iter    = 50;    %����������
N_pop       = 100;   %��Ⱥ��������
%

if n_sence == 1
    Price_change = [0, 0, 0, 0, 0];   %��Ǽ۸�仯��0���䣬1�仯,[0, 1, 1, 0, 0]
    % �۸�仯��Χ��������һ�����ޣ��ڶ������ޣ��������ɼ۸�Ĳ�ȷ����Χ
    Price_range = [1, 1; 1.2049, 1; 1.1852, 0.8148; 1, 1; 1, 1]';  
    Price_dist  = 1;   % �۸�ֲ���1Ϊ������ɣ�2Ϊ���ȷֲ�
    % ��������ȷ���ԣ�����Ĭ��Ϊ0.1ʱ����ʾ�����ڻ�׼ֵ��90%-110%��Χ
    Uncertain_source = 0.0;
    % ���ɲ�ȷ���ԣ�����Ĭ��Ϊ0.5ʱ����ʾ�����ڻ�׼ֵ��100%-200%��Χ
    Uncertain_load   = 0.0; 
else
    Price_change = [0, 1, 1, 0, 0];   %��Ǽ۸�仯��0���䣬1�仯,[0, 1, 1, 0, 0]
    % �۸�仯��Χ��������һ�����ޣ��ڶ������ޣ��������ɼ۸�Ĳ�ȷ����Χ
    Price_range = [1, 1; 1.2049, 1; 1.1852, 0.8148; 1, 1; 1, 1]';  
    Price_dist  = 1;   % �۸�ֲ���1Ϊ������ɣ�2Ϊ���ȷֲ�
    % ��������ȷ���ԣ�����Ĭ��Ϊ0.1ʱ����ʾ�����ڻ�׼ֵ��90%-110%��Χ
    Uncertain_source = 0.1;   % Ĭ��0.1���û����޸�
    % ���ɲ�ȷ���ԣ�����Ĭ��Ϊ0.5ʱ����ʾ�����ڻ�׼ֵ��100%-200%��Χ
    Uncertain_load   = 0.5;   % Ĭ��0.5���û����޸�
end

N_obj      = 1;      %�Ż�Ŀ����
dim        = 24*Alc_year;
% ����ά�ȣ�ǰ24ά��ʱ���ʣ�25-48���ʵ�ʳ�����49ά��������
if N_obj == 2
    x3_limt   = [10, 18];          %���ܽ���Ľڵ�����
    % �������ܽṹ����������
    % ��һ�У��ڵ�ţ��ڶ��У������й���kW���������У������޹�(kvar)
    Bus    = struct2array(load('Bus_IEEE33.mat'));
    % ��1�д�֧·�ţ���2�д�֧·�׽ڵ㣬��3�д�֧·β�ڵ㣬��4��֧·���裬��5�д�֧·�翹 ��ŷķ��
    Branch = struct2array(load('Branch_IEEE33.mat'));
end

%% ����۸�
sell_price  = ones(1, 24) * 0.538;   %��λ��0.488 Ԫ/kWh
lease_price = ones(1, 24) * 108;     %�����������浥�ۣ�108 Ԫ/kWh
data_price  = xlsread('Data_Input.xlsx', 'Price');    %�۸�
buy_price   = data_price(:,1)';   %��λ��Ԫ/kWh
dr_price    = data_price(:,4)';   %������Ӧ�۸�2 Ԫ/kWh
model_price = 0;                  %����������۸�Ԫ/kWh
Price       = [buy_price; sell_price; lease_price; dr_price; model_price*ones(1,24)];

%% ��������Դ�滮
% ���װ������������Դ�����������÷�������_2024.10.21��
Source_Plan = xlsread('Data_Input.xlsx', '���滮', 'B3:C27')';   % B2:C26

% *********************���������뼰�䲻ȷ��������*********************** %
Capacity_WT  = Source_Plan(1, :);   %���װ��������45��ǧ�ߣ�26��ײ���
Capacity_PV  = Source_Plan(2, :);   %���װ��������47.5��ǧ�ߣ�25��ײ���
if strcmpi(season, '��') == 1
    Source_coef = xlsread('Data_Input.xlsx', '���ϵ��', 'A2:B25')';   %�ļ�����ϵ����������
else
    Source_coef = xlsread('Data_Input.xlsx', '���ϵ��', 'C2:D25')';   %��������ϵ����������
end
% ����ϵ�����ݡ��ӳص��������Ŀ15�����ӳ���(������ʢ��������)������
for i = 1:Alc_year
    WT(i,:) = 10000*Capacity_WT(i).*Source_coef(1,:);   %����������λ����ǧ��ת��kW
    PV(i,:) = 10000*Capacity_PV(i).*Source_coef(2,:);   %�����������λ����ǧ��ת��kW
    Renewable(i,:) = PV(i,:) + WT(i,:);                 %����Դ=���+��磬������
end
WT        = WT .* Rate_Power;
PV        = PV .* Rate_Power;
Renewable = Renewable .* Rate_Power;                    %����������
% ---------------------���������뼰�䲻ȷ���Խ���----------------------- %

% ***********************�������뼰�䲻ȷ��������************************* %
% ������Դ�����ӳ��ϵ�����룩��ҵ԰Դ���ɴ�һ�廯��Ŀǰ�ڸ��ɵ��鱨�棨10�·ݣ� (11.11-y��v3)��
% ���ݷ��������㸺�ɣ����ճ������������ɵ���=1��1.5Ϊ��׼����
Load_access = ones(1, 1);   %���ɽ���,����8��1��ʾ���룬0��ʾ�����룬
Uncertainty = '��̬';  %��ȷ���Էֲ�����̬�ֲ�����ȷֲ�

if length(Load_access) == 1
    %���ݳ����͸�����ȼ��㸺��
    Load_Coef = xlsread('Data_Input.xlsx', '���и���', 'B7:C30');   %����ϵ��
    if strcmpi(season, '��') == 1
        Load_coef = Load_Coef(:, 2:2:end);     %�ļ�����ϵ��
        Load_max  = ones(Alc_year, 1) .* sum(Renewable,2) / sum(Load_coef) /10000 /0.85;
    else
        Load_coef = Load_Coef(:, 1:2:end-1);   %��������ϵ��
        Load_max  = ones(Alc_year, 1) .* sum(Renewable,2) / sum(Load_coef) /10000 /0.85;
    end   
else
    Load_max  = xlsread('Data_Input.xlsx', '����', 'B4:Q5');    %��󸺺�
    Load_Coef = xlsread('Data_Input.xlsx', '����', 'B7:Q30');   %����ϵ��
    if strcmpi(season, '��') == 1
        Load_max  = Load_max(:, 2:2:end);      %�ļ���󸺺�
        Load_coef = Load_Coef(:, 2:2:end);     %�ļ�����ϵ��
    else
        Load_max  = Load_max(:, 1:2:end-1);    %������󸺺�
        Load_coef = Load_Coef(:, 1:2:end-1);   %��������ϵ��
    end
end
Load_Max  = sum(Load_max, 2);
Load_plan = sum(Load_access.*Load_max, 2).*0.85;   %25���Ժ�ĸ��ɹ滮ֵ.0.85Ϊͬʱ��
for i = 1:Alc_year
    P_load(i,:) = 10000*sum(Load_access.* Load_max(i, :).*Load_coef, 2)'.*0.85;
    P_load(i,:) = P_load(i,:) .* Rate_Load;   %
    Rate_Load_cal(i) = sum(P_load(i,:)) / sum(Renewable(i,:));
end
% -----------------------�������뼰�䲻ȷ���Խ���------------------------- %

SOC_0       = 0.3;   %����ϵͳ��ʼSOC
SOC_limt    = [0.1; 1];     %SOCԼ����10%-100%
%% ���Ǹ������صĴ��������Ż�,ʤ��Դ�ɲ�ȷ�����泡��
% ����3������ԭ�����ݷֲ���-+3sigma��Χ�ڵĸ���Ϊ99.73%
% ��˸��ݲ�ȷ����Χ�����������ͳ�Ʒֲ��ı�׼��

N_s    = 1000*ceil(n_sence/100);   %���ɳ�����
N_sence = 100*ceil(n_sence/100);   %������������������100��
% LHS����Դ�ɲ�ȷ���Եķ��泡��
Load_all = [];   %1000*(24*Alc_year)���ո��ɷ��泡��
EnWT_all = [];   %1000*(24*Alc_year)���շ��������泡��
EnPV_all = [];   %1000*(24*Alc_year)���չ���������泡��
Ener_all = [];   %1000*(24*Alc_year)���շ��������泡��
if strcmpi(Uncertainty, '��̬') == 1
    for k = 1:Alc_year
        be = (k-1)*24;
%         en = k*24;
        for t = 1:24
            mu_l(t)     = P_load(k,t)*1;
            sigma_l(t)  = P_load(k,t)*Uncertain_load/3;

            mu_wt(t)    = WT(k,t);
            sigma_wt(t) = WT(k,t)*Uncertain_source/3;

            mu_pv(t)    = PV(k,t);
            sigma_pv(t) = PV(k,t)*Uncertain_source/3;       

            load_temp   = sort(lhsnorm(0, 1, N_s), 'descend');
            Load_temp   = load_temp .* sigma_l(t) + mu_l(t);
            ener_temp   = sort(lhsnorm(0, 1, N_s), 'descend');
            EnWT_temp   = ener_temp .* sigma_wt(t) + mu_wt(t);
            EnPV_temp   = ener_temp .* sigma_pv(t) + mu_pv(t);
            Ener_temp   = EnWT_temp + EnPV_temp;

            Result_mu(be+t, 1) = mean(Load_temp);
            Result_mu(be+t, 2) = mean(Ener_temp);
            Result_sigma(be+t, 1) = std(Load_temp);
            Result_sigma(be+t, 2) = std(Ener_temp);

            Load_all(:,be+t) = sort(Load_temp, 'descend');
            EnWT_all(:,be+t) = sort(EnWT_temp, 'descend');
            EnPV_all(:,be+t) = sort(EnPV_temp, 'descend');
            Ener_all(:,be+t) = sort(Ener_temp, 'descend');
        end
    end
else
    for k = 1:Alc_year
        be = (k-1)*24;
%         en = k*24;
        for t = 1:24
            load_max = P_load(k,t)*(1+Uncertain_load);
            load_min = P_load(k,t)*(1-Uncertain_load);

            wt_max   = WT(k,t)*(1+Uncertain_source);
            wt_min   = WT(k,t)*(1-Uncertain_source);

            pv_max   = PV(k,t)*(1+Uncertain_source);
            pv_min   = PV(k,t)*(1-Uncertain_source);

            Load_temp = linspace(load_max, load_min, N_s)';
            EnWT_temp = linspace(wt_max, wt_min, N_s)';
            EnPV_temp = linspace(pv_max, pv_min, N_s)';
            Ener_temp = EnWT_temp + EnPV_temp;

            Load_all(:,be+t) = sort(Load_temp, 'descend');
            EnWT_all(:,be+t) = sort(EnWT_temp, 'descend');
            EnPV_all(:,be+t) = sort(EnPV_temp, 'descend');
            Ener_all(:,be+t) = sort(Ener_temp, 'descend');            
        end
    end
end
% �����²������ٳ���������1000������100
Load_All = [];   %100*(24*Alc_year)��ÿ��Ϊ�ո���ʱ������
EnWT_All = [];   %100*(24*Alc_year)��ÿ��Ϊ�շ�����ʱ������
EnPV_All = [];   %100*(24*Alc_year)��ÿ��Ϊ�չ������ʱ������
Ener_All = [];   %100*(24*Alc_year)��ÿ��Ϊ�ճ���ʱ������
for i = 1:N_sence
    if i <= (N_sence/2)
        po = (i-1)*10 + ceil(i*10/N_sence) + 1;
        Load_All(i,:) = Load_all(po,:);
        EnWT_All(i,:) = EnWT_all(po,:);
        EnPV_All(i,:) = EnPV_all(po,:);
        Ener_All(i,:) = Ener_all(po,:);
    else
        po = (i-1)*10 + ceil(i*10/N_sence) - 1;
        Load_All(i,:) = Load_all(po,:);
        EnWT_All(i,:) = EnWT_all(po,:);
        EnPV_All(i,:) = EnPV_all(po,:);
        Ener_All(i,:) = Ener_all(po,:);
    end 
end

%% PSO�㷨����
% ����ά�ȣ�ǰ24ά��ʱ���ʣ�25-48���ʵ�ʳ�����49ά��������
% �˴�����Ϊ���ܷų��Ĺ��ʣ�+�����͸����ܳ��Ĺ��ʣ�-��
% ���ܵ���ڵĵ����仯���ŵ�Ϊ����*1/0.93��+�������Ϊ����*1*0.93��-��

w_max      = 0.9;   %������ϵ��
w_min      = 0.4;   %��С����ϵ��
v1_max(1, 1:24) = Storage_min*0.1*1000*ones(1,24);     %�ٶ��Ͻ�
v1_max = repmat(v1_max, 1, Alc_year);
v2_max(1, 1:24) = max(max(Renewable))*0.1*ones(1,24);
v2_max = repmat(v2_max, 1, Alc_year);
v3_max(1,1)  = Storage_min*0.5*1000;
if N_obj == 2
    v3_max(1,2) = ceil((x3_limt(2)-x3_limt(1))/5);
end
v1_min = -v1_max;
v2_min = -v2_max;
v3_min = -v3_max;

%% ���Ǹ������صĴ��������Ż�������ÿ�������µ����Ŵ��ܣ�
Result_All = {};
Grid_All   = {};
Result_Opt = [];
Grid_Opt   = [];
Fitness    = [];

Storage_max = Storage_max*1000;   %��λ��MWhת��kWh
Storage_min = Storage_min*1000;   %��λ��MWhת��kWh
for ns = 1:n_sence   % ns = 1:n_sence
    renewable_ns = Ener_All(ns, :);   %��ǰ����������
    pload_ns     = Load_All(ns, :);   %��ǰ��������
    Price_ns     = repmat(Price, 1, Alc_year);   %���ݼ۸��Ƿ񲨶��趨�����ɵ�ǰ�����۸�
    X1_upper = [];   X1_lower = [];
    X2_upper = [];   X2_lower = [];
    X3_upper = [];   X3_lower = [];
    for k = 1:length(Price_change)
        if Price_change(k) == 1
            if Price_dist == 1
                price_h = Price(k,1)*Price_range(1,k);
                price_l = Price(k,1)*Price_range(2,k);
                Price_ns(k,1) = rand(1) * (price_h - price_l) + price_l;
            else
                price_temp = linspace(Price(k,1)*Price_range(2,k), Price(k,1)*Price_range(1,k), n_sence);
                Price_ns(k,1) = price_temp(ns);
            end
        end
    end
    Price_Ns(:, ns) = Price_ns(:, 1);

    %% PSOѭ��
    for i = 1:24
        if sell_price(i) > buy_price(i)
            X1_upper(1,i) = 0 ;           %ǧ�ߣ���ŵ����0.5C���ŵ�Ϊ(+)
            X1_lower(1,i) = -Storage_min/2;   %ǧ�ߣ���ŵ����0.5C�����Ϊ(-)
        else
            X1_upper(1,i) = Storage_min/2 ;   %ǧ�ߣ���ŵ����0.5C���ŵ�Ϊ(+)
            X1_lower(1,i) = -Storage_min/2;   %ǧ�ߣ���ŵ����0.5C�����Ϊ(-)      
        end
    end
    X1_upper = repmat(X1_upper, 1, Alc_year);
    X1_lower = repmat(X1_lower, 1, Alc_year);

    for i = 1:24*Alc_year
        X2_upper(1,i) = renewable_ns(i);
        if renewable_ns(i) <= pload_ns(i)
            X2_lower(1,i) = renewable_ns(i);
        else
            X2_lower(1,i) = Rate_limt(1)*renewable_ns(i);
        end
    end
    
    X3_upper(1,1) = Storage_max;
    X3_lower(1,1) = Storage_min;
    if N_obj == 2
        X3_upper(1,2) = x3_limt(2);
        X3_lower(1,2) = x3_limt(1); 
    end
    
    X1 = Func_initialization(N_pop, X1_upper, X1_lower, dim);   %��ʼ����Ⱥ������
    X2 = Func_initialization(N_pop, X2_upper, X2_lower, dim);   %��ʼ����Ⱥ�����ʵ�ʳ���
    X3 = Func_initialization(N_pop, X3_upper, X3_lower, N_obj); %��ʼ����Ⱥ��������λ��
    X3(:, 1) = round(X3(:, 1)/10000)*10000;   %��������С�߶�10MW
    if N_obj == 2
        X3(:, 2) = round(X3(:, 2));           %λ��
    end
    
    V1 = Func_initialization(N_pop, v1_max, v1_min, dim);   %��ʼ����Ⱥ�ٶ�
    V2 = Func_initialization(N_pop, v2_max, v2_min, dim);   %��ʼ����Ⱥ�ٶ�
    V3 = Func_initialization(N_pop, v3_max, v3_min, N_obj);   %��ʼ����Ⱥ�ٶ�
    
    for i = 1:N_pop
        Plim = pload_ns - X2(i,:);   %���ܷŵ����ƣ�������������͵�
        [X1(i,:), X3(i,1)] = Func_ConstraintCheck(X1(i,:), X3(i, 1), SOC_limt, SOC_0, Plim);
    end

    fx = zeros(N_obj, N_pop);
    % ��������Ŀ�꺯����1�����ܾ����ԣ�ȡ���2���ڵ��ѹ����������ȡ���
    % ��������Ŀ�꺯��1�����ܾ����ԣ�ȡ���
    for i = 1:N_pop
        Storage = X3(i, 1);
        fx(1, i) = Func_obj(X1(i,:), X2(i,:), X3(i, 1), pload_ns, Price_ns, renewable_ns);   %������Ӧ��ֵ
    end     
    % ��������Ŀ�꺯��2���ڵ��ѹ����������ȡ���
    if N_obj == 2
        V_bus = [];   % 24��
        for k = 1:24
            P_new = [EnWT_All(ns, k), EnPV_All(ns, k)];
            Location_Storage = X(i, 50);   %����λ��
            v_bus = Func_power_flow(X(i, k), Location_Storage, pload_ns(k), P_new, Bus, Branch);
            V_bus = [V_bus, v_bus];
        end
        for k = 1:size(V_bus, 1)
            vbus = V_bus(k, :);
            temp_fx2(k) = sum(abs(vbus-mean(vbus)));
        end
        fx(2, i)  = 1/sum(temp_fx2);   % 1/ƽ���ڵ��ѹ������
    end
    
    pBest1   = X1;   % ����ʼ��Ⱥ��Ϊ��ʷ����
    pBest2   = X2;
    pBest3   = X3;
    fx_pBest = fx;   % ��¼��ʼȫ�����Ž�,Ĭ���Ż����ֵ

    % ��¼��ʼȫ�����Ž�
    [~, index]  = max(sum(fx, 1));   %Ѱ����Ӧ������λ��    
    fx_gBest    = -inf;   %��¼��Ӧ��ֵ��λ��
    gBest1      = X1(index(1),:);
    gBest2      = X2(index(1),:);
    gBest3      = X3(index(1),:);
    gBest_index = index(1);

    X1new     = X1;             % ��λ��
    X2new     = X2;             % ��λ��
    X3new     = X3;             % ��λ��
    fx_New    = fx;              % ��λ����Ӧ��ֵ
    
    %% ***********************PSOѭ����ʼ************************* %
    for t = 1:Max_Iter
        wait = ceil(100*ns/n_sence)-1 + ceil(100*t/Max_Iter)/100;
        wait_str = ['���泡��ѭ����� ', num2str(wait), '%'];
        waitbar(wait/100, hwait, wait_str);
        % ������Ⱥ��Ӧ��
        for i = 1:N_pop
            w  = w_max-(w_max-w_min)*(t)^2/(Max_Iter)^2;   %����Ȩ�ظ���
            c1 = (0.5-2.5)*t/Max_Iter+2.5;                 %��������c1����
            c2 = (2.5-0.5)*t/Max_Iter+0.5;                 %��������c2����
            r1 = rand(1,dim);
            r2 = rand(1,dim);
            r3 = rand(1,N_obj);
            r4 = rand(1,N_obj);           
            V1(i,:) = w*V1(i,:) + c1.*r1.*(pBest1(i,:) - X1(i,:)) + c2.*r2.*(gBest1 - X1(i,:));
            V2(i,:) = w*V2(i,:) + c1.*r1.*(pBest2(i,:) - X2(i,:)) + c2.*r2.*(gBest2 - X2(i,:));
            V3(i,:) = w*V3(i,:) + c1.*r3.*(pBest3(i,:) - X3(i,:)) + c2.*r4.*(gBest3 - X3(i,:));
            V1(i,:) = Func_BoundaryCheck(V1(i,:), v1_max, v1_min, dim);   %�ٶȱ߽��鼰Լ��
            V2(i,:) = Func_BoundaryCheck(V2(i,:), v2_max, v2_min, dim);   %�ٶȱ߽��鼰Լ��
            V3(i,:) = Func_BoundaryCheck(V3(i,:), v3_max, v3_min, N_obj);   %�ٶȱ߽��鼰Լ��
            X1new(i,:) = X1(i,:) + V1(i,:);                           %λ�ø���
            X2new(i,:) = X2(i,:) + V2(i,:);
            X3new(i,:) = X3(i,:) + V3(i,:);
            
            % ----------------���ӱ߽��⼰Լ��------------------ %
            X2new(i,:) = Func_BoundaryCheck(X2new(i,:), X2_upper, X2_lower, dim);
            X3new(i,:) = Func_BoundaryCheck(X3new(i,:), X3_upper, X3_lower, N_obj);
            if N_obj == 2
                X3new(i,2) = round(X3new(i,2));   %����λ��
            end
            X3new(i, 1)  = round(X3new(i, 1)/10000)*10000;   %��������С�߶�10MW
            
            Storage = X3new(i, 1);
            Plim    = pload_ns - X2new(i,:);   %���ܷŵ����ƣ�������������͵�
            [X1new(i,:), X3new(i,1)] = Func_ConstraintCheck(X1new(i,:), Storage, SOC_limt, SOC_0, Plim);

            Grid(:, i) = (pload_ns - (X1new(i,:) + X2new(i,:)))'./1000;
            if min(Grid(:, i)) < 0
                Grid_index(1, i) = -1;
            else
                Grid_index(1, i) = 1;
            end
            
            % ��������Ŀ�꺯����1�����ܾ����ԣ�ȡ���2���ڵ��ѹ����������ȡ���
            % ��������Ŀ�꺯��1�����ܾ����ԣ�ȡ���
            for i = 1:N_pop
                Storage = X3(i, 1);
                fx(1, i) = Func_obj(X1(i,:), X2(i,:), X3(i, 1), pload_ns, Price_ns, renewable_ns);   %������Ӧ��ֵ
            end     
            % ��������Ŀ�꺯��2���ڵ��ѹ����������ȡ���
            if N_obj == 2
                V_bus = [];   % 24��
                for k = 1:24
                    P_new = [EnWT_All(ns, k), EnPV_All(ns, k)];
                    Location_Storage = X(i, 50);   %����λ��
                    v_bus = Func_power_flow(X(i, k), Location_Storage, pload_ns(k), P_new, Bus, Branch);
                    V_bus = [V_bus, v_bus];
                end
                for k = 1:size(V_bus, 1)
                    vbus = V_bus(k, :);
                    temp_fx2(k) = sum(abs(vbus-mean(vbus)));
                end
                fx(2, i)  = 1/sum(temp_fx2);   % 1/ƽ���ڵ��ѹ������
            end
            
             % ������ʷ����ֵ��Ŀ�꺯����Ȩ��
            if sum(fx_New(:, i)) >= sum(fx_pBest(:, i))
                pBest1(i,:)    = X1new(i,:);
                pBest2(i,:)    = X2new(i,:);
                pBest3(i,:)    = X3new(i,:);
                fx_pBest(:, i) = fx_New(:, i);    
            end
            % ����ȫ������ֵ������Ŀ�꺯����Ȩ��
            if sum(fx_New(:, i)) >= sum(fx_gBest)
                fx_gBest = fx_New(:, i);
                gBest1   = X1new(i,:);
                gBest2   = X2new(i,:);
                gBest3   = X3new(i,:);
                gBest_index = i;
            end 
        end

        X1 = X1new;
        X2 = X2new;
        X3 = X3new;
        fx = fx_New;
        % ��¼��ǰ��������ֵ��������Ӧ��ֵ
        IterCurve(t, :) = fx_gBest;    %��¼��ǰ���������Ž���Ӧ��ֵ
    end
    %% -----------------------PSOѭ������------------------------- %        
    Result_All{1, ns} = X1./1000;
    Result_All{2, ns} = X2./1000;
    Result_All{3, ns} = X3./1000;
    
    temp_gBest1 = gBest1'./1000;
    temp_gBest2 = gBest2'./1000;
    temp_gBest3 = gBest3'./1000;
    if N_obj == 2
        temp_gBest3(:,2).*1000;
    end
    
    Max_Pdelta = max(renewable_ns - pload_ns)/1000;  % ������-��������ֵ��MW
    Max_Pstor = max(abs(temp_gBest1));               % ����ŵ繦�ʣ�MW
    Rate = sum(temp_gBest2)/sum(renewable_ns./1000);  %�����ʣ�%
    Elec_rate = sum(temp_gBest2)/sum(pload_ns/1000);  %����Դ����ռ�ȣ�%
    grid_index = Grid_index(gBest_index);
    Result_Opt(:, ns) = [temp_gBest3; ns; Max_Pdelta; Max_Pstor; Rate; Elec_rate; grid_index];
    Result_Opt1(:, ns) = temp_gBest1;
    Result_Opt2(:, ns) = temp_gBest2;
    
    Grid_Opt(:, ns) = Grid(:, gBest_index);
    if N_obj == 1
        Fitness(:, ns)  = IterCurve;
    else if N_obj == 2
            Fitness(:, ns)  = [IterCurve(:, 1); IterCurve(:, 2)];
        end
    end
    
end
toc;
close(hwait);

%% 
Result_sorted = sortrows(Result_Opt', 1);
if ns == 1
    opt_ns = ns;
else
    opt_ns = floor(ns*0.5);
end
Opt_ns = Result_sorted(opt_ns, N_obj+1);
Grid_opt = Grid_Opt(:, Opt_ns);

%% ������
P_stor_opt  = Result_Opt1(1:end, Opt_ns)';     %����24Сʱ���ʣ�kW
disp('���������Ż����������MW/����MWh')
max_Pstor = max(abs(P_stor_opt));
Storage_opt = Result_Opt(1, Opt_ns);
[max_Pstor, Storage_opt]
disp('����-����������������������MWh��')
max_Pdelta = max(Ener_All(Opt_ns, :)/1000 - Load_All(Opt_ns, :)/1000);
plim = Load_All(Opt_ns, :) - Ener_All(Opt_ns, :);
Sum_Pchr = -sum(plim(find(plim<0)))/1000/Alc_year;
[max_Pdelta, Sum_Pchr]
disp('�����ʣ�����Դ����ռ�ȣ�%�����õ�����������������������ǧ��ʱ��')
rate      = sum(Result_Opt2(:, Opt_ns))/sum(Ener_All(Opt_ns, :)/1000);
elec_rate = sum(Result_Opt2(:, Opt_ns))/sum(Load_All(Opt_ns, :)/1000);
Elec_generation  = sum(Result_Opt2(:, Opt_ns)*1000)*330/100000000/Alc_year;
Elec_consumption = sum(sum(P_load))*330/100000000/Alc_year;
Elec_buy = Elec_consumption - Elec_generation;
[rate, elec_rate, Elec_consumption, Elec_generation, Elec_buy]
Result = Result_All{1, Opt_ns};


%% �Ż������ͼ
be = 1;
en = 24;
figure(1)
stairs(Load_All(Opt_ns, be:en)/1000,  'Color', 'b', 'LineWidth', 2)
hold on
stairs(Ener_All(Opt_ns, be:en)/1000, 'Color', 'g', 'LineWidth', 2)
hold on
line([0, 24*Alc_year], [mean(Load_All(Opt_ns, :))/1000, mean(Load_All(Opt_ns, :))/1000], 'linestyle','--', 'color', 'b')
text(24*Alc_year+0.2, mean(Load_All(Opt_ns, :)/1000), num2str(mean(Load_All(Opt_ns, :))/1000))
hold on
line([0, 24*Alc_year], [mean(Ener_All(Opt_ns, :))/1000, mean(Ener_All(Opt_ns, :))/1000], 'linestyle','--', 'color', 'g')
text(24*Alc_year+0.2, mean(Ener_All(Opt_ns, :))/1000, num2str(mean(Ener_All(Opt_ns, :))/1000))
legend('����', '������')
xlim([be-1, en])
xlabel('t/h')
ylabel('MW')


figure(2)
% ���������¶�Ӧ���չ�������
stairs(P_stor_opt(be:en), '-s','Color',[1 0.64706 0], ...
     'MarkerEdgeColor', 'k','MarkerFaceColor','r','LineWidth', 2)
hold on
line([0,24*Alc_year], [0,0],'linestyle','--','color','r')
xlim([be-1, en])
xlabel('t/h')
ylabel('����/MW')
title('���ܳ�ŵ繦��')

figure(3)
if N_obj == 1
    fitness = Fitness(:, opt_ns)';
    t = 1:length(fitness);
    fitness(find(fitness<-1000)) = 0;
    plot(t, fitness)
    % ylim([-3, 3])
    xlabel('��������')
    ylabel('�Ż�Ŀ��')
else if N_obj == 2
        subplot(211)
        fitness = Fitness(1:Max_Iter, opt_ns)';
        t = 1:length(fitness);
        fitness(find(fitness<-1000)) = 0;
        plot(t, fitness)
        % ylim([-3, 3])
        xlabel('��������')
        ylabel('�Ż�Ŀ��1')
        
        subplot(212)
        fitness = Fitness(Max_Iter+1:Max_Iter*2, opt_ns)';
        t = 1:length(fitness);
        fitness(find(fitness<-1000)) = 0;
        plot(t, fitness)
        % ylim([-3, 3])
        xlabel('��������')
        ylabel('�Ż�Ŀ��2')
    end
end


figure(4)
x    = P_stor_opt(1:24);
soc0 = Storage_opt*SOC_0;
SOC  = [soc0];
for i = 1:24
    if x(i) > 0        %�ŵ�
        w_stor(i) = -x(i)*1/0.93;   %x(i)>0���ŵ磬��ص�������w(i),-
    else 
        w_stor(i) = -x(i)*1*0.93;   %x(i)<0����磬��ص�������w(i),+
    end
end
for i = 1:24
    soc = soc0 + sum(w_stor(1:i));
    SOC = [SOC; soc];
end
SOC = SOC ./ Storage_opt;
plot([0:24], SOC, 'Color', 'b', 'LineWidth', 2)
xlim([0, 24])
title('SOC')
xlabel('t/h')
ylabel('SOC')

figure(5)
po_g1 = find(Result_Opt(N_obj+6,:) == 1);
po_g2 = find(Result_Opt(1,po_g1) < (Storage_max/1000));
if length(Result_Opt(1,po_g1(po_g2))) > 0
    histogram(Result_Opt(1,po_g1(po_g2)), 15)
    hold on
    line([Storage_opt, Storage_opt], [0, 30], 'linestyle','--','color','r','linewidth',2)
    hold on
    text(Storage_opt, ns*0.15, strcat(num2str(Storage_opt), 'MWh' ))
    hold on
    if length(po_g1) > 0
        text(max(Result_Opt(1,po_g1))*1.05, 1, strcat(num2str(length(po_g1)) ))
    end
    xlabel('MWh')
    ylabel('Ƶ��')
    title('Ƶ���ֲ�ֱ��ͼ')
end

figure(6)
stairs(Grid_opt, '-s','Color',[1 0.64706 0], ...
     'MarkerEdgeColor', 'k','MarkerFaceColor','r','LineWidth', 2)
hold on
line([0,24*Alc_year], [0,0],'linestyle','--','color','r')
% xlim([0, 24])
xlabel('t/h')
ylabel('����/MW')
title('��������')


%% ����Ϊ����

%% ����Ⱥ��ʼ������,��ʼ��λ�û��ٶ�
function X = Func_initialization(pop, ub, lb, dim)
    % pop:Ϊ��Ⱥ����
    % dim:ÿ������Ⱥ��ά��
    % ub: Ϊÿ��ά�ȵı����ϱ߽磬ά��Ϊ[1,dim];
    % lb: Ϊÿ��ά�ȵı����±߽磬ά��Ϊ[1,dim];
    % X:  Ϊ�������Ⱥ��ά��[pop,dim];
    X = zeros(pop, dim); %ΪX���ȷ���ռ�
    for i = 1:pop
       for j = 1:dim
           X(i,j) = (ub(j) - lb(j))*rand() + lb(j);  %����[lb,ub]֮��������
       end
    end
end


%% �߽��麯��
function X = Func_BoundaryCheck(x, ub, lb, dim)
    % dimΪ���ݵ�ά�ȴ�С
    % xΪ�������ݣ�ά��Ϊ[1,dim];
    % ubΪ�����ϱ߽磬ά��Ϊ[1,dim]
    % lbΪ�����±߽磬ά��Ϊ[1,dim]
    if sum(isnan(x)) > 0
        for i = 1:dim
            x(i) = (ub(i)+lb(i))/2
        end
    else       
        for i = 1:dim
            if x(i) > ub(i)
               x(i) = ub(i); 
            end
            if x(i) < lb(i)
                x(i) = lb(i);
            end
        end
    end
    X = x;
end


%% ���ܳ�ŵ繦��Լ������
function [X1new, X3new] = Func_ConstraintCheck(X, Storage, SOC_limt, SOC_0, Plim)
    % dimΪ���ݵ�ά�ȴ�С
    % xΪ�������ݣ�ά��Ϊ[1,dim];
    % SOC_limt��SOC���½�Լ����[0.1, 1]
    % Rate_limt�����������½�Լ����[0.95, 1]
    SOC     = [];
    SOC_re  = [];
    SOC_min = Storage*SOC_limt(1);
    SOC_max = Storage*SOC_limt(2);
    X_3 = Storage;
    eta_chg = 0.927;   % ���Ч��
    eta_dc  = 0.928;   % �ŵ�Ч��
    % ��ŵ�Լ��˼·������ŵ繦��ת������صĵ����仯w�����ӻ���٣�����������Ϊ0��
    % w���ۼӼ���SOC��w������Լ����ŵ�ʱ�̡�
    
    for k = 1:length(X)/24
        % Լ��ÿ�ճ�ŵ������
        ii1 = 24 *(k-1) + 1;
        ii2 = 24 * k;
        x = X(ii1:ii2);
        for i = 1:24
            if x(i) >= 0        %�ŵ�
                w_stor(i) = -x(i)*1/eta_dc;   %x(i)>0���ŵ磬��ص�������w(i)<0
            else if x(i) < 0   %���
                w_stor(i) = -x(i)*1*eta_chg;   %x(i)<0����磬��ص�������w(i)>0
                end
            end
        end
        w_stor = w_stor - (sum(w_stor)-0)/24;

        % buy_price<sell_price��������,[1,2,3,4,5,6,7,24]ʱ�̲�����磬x��ҪС��0
        for i= 1:7
            if w_stor(i) <= 0   %�ŵ磬�������٣�w(i)<0
                temp1 = Storage*0.01*rand()*eta_dc;
                w_stor(8:23) = w_stor(8:23) - (temp1 - w_stor(i))/16;
                w_stor(i) = temp1;
            end
        end
        if w_stor(24) <= 0     %�ŵ磬�������٣�w(i)<0
            temp2 = Storage*0.01*rand()*eta_dc;
            w_stor(8:23) = w_stor(8:23) - (temp2 - w_stor(i))/16;
            w_stor(24) = temp2;
        end

        % ��ŵ�����Լ�������ܳ���Storage*0.5
        pro_x = zeros(24, 1);     %�����仯w�ͳ�ŵ繦��Լ���µĵ����仯��ֵ������1�����ʳ���Լ��
        for i = 1:24
            if w_stor(i) <= 0   %�ŵ磬��������
                pro_x(i) = w_stor(i)/(-Storage*0.5*1/eta_dc);
            else if w_stor(i) > 0   %��磬��������
                    pro_x(i) = w_stor(i)/(Storage*0.5*1*eta_chg);
                end
            end
        end
        if max(pro_x) > 1   %����1�����ŵ繦���г���Լ����ʱ��
            w_stor = w_stor .* (1/max(pro_x));
        end

        % SOCԼ����min<SOC��t��<max��t\in[1:24]
        soc0   = Storage*SOC_0;
        for i = 1:24
            soc = soc0 + sum(w_stor(1:i));
            SOC = [SOC; soc];
        end
        soc_pro1 = (Storage*(1-SOC_0))/(max(SOC)-soc0);
        soc_pro2 = (Storage*(SOC_0-0.1))/(soc0-min(SOC)); 
        w_stor = w_stor .* min(soc_pro1, soc_pro2);

        % ��ֹ���͵�Լ��������-����ԴΪ���ƹ��ʣ����ܷŵ�������ƣ������ַ��͵�
        po_lim  = [];
        po_char = find(w_stor > 0);   %���ʱ��
        po_disc = find(w_stor <= 0);  %�ŵ�ʱ��
        w_sum   = 0;
        for i = 1:24
            % ������ڸ���&�����ڷŵ磬�᷵�͵磬��Ϊ���
            if Plim(i)<0 & w_stor(i)<0
                po_lim = [po_lim; i];                  %���͵�ʱ��
                delta_w = -Plim(i)*eta_dc - w_stor(i);   %����������>0
                w_sum = w_sum + abs(delta_w);             %�ŵ�ת��磬��������
                w_stor(i) = -Plim(i)*eta_chg;             %�෢�ĵ綼�������
            % ������ڸ���&�����ڳ�磬����繦�ʲ��㣬�᷵�͵磬���ӳ�繦��
            else if Plim(i)<0 & -w_stor(i)/eta_chg > Plim(i)
                    po_lim = [po_lim; i];                  %���͵�ʱ��
                    delta_w = -Plim(i)*eta_chg - w_stor(i);   %����������>0
                    w_sum = w_sum + abs(delta_w);             %�������������
                    w_stor(i) = -Plim(i)*eta_chg;             %�෢�ĵ綼�������
                % ����С�ڸ���&�����ڷŵ磬���ŵ繦�ʹ��󣬻᷵�͵磬���ٷŵ繦��
                else if Plim(i)>0 & -w_stor(i)*eta_dc > Plim(i)
                        po_lim = [po_lim; i];                  %���͵�ʱ��
                        delta_w = -Plim(i)/eta_dc - w_stor(i);   %����������>0
                        w_sum = w_sum + abs(delta_w);             %�ŵ��С����������
                        w_stor(i) = -Plim(i)/eta_dc;             %���ٷŵ�
                    end
                end
            end
        end
        % �������͵�ʱ�̹��ʣ����·��͵�ʱ���������ӣ�����ʱ����Ҫ���ٳ����
        if length(po_lim) > 0
            po2 = [1:7];
            char_sum = sum(w_stor(po2));       %����ǰ�����������
            delta = abs(char_sum) - abs(w_sum);
            if delta > 0
                pro_lim = delta / abs(char_sum);
                w_stor(po2) = w_stor(po2).*pro_lim;
            else
                w_stor(po2) = 0;    %
                po3 = [8:23];
                po4 = setdiff(po3, po_lim);
                disc = delta / length(po4);  %20250108�޸ģ�����������ŵ���󣬵��·��͵�
                delta_k = 0;
                for i = 1:length(po4)
                    j = po4(i);
                    if -(w_stor(j)+disc+delta_k)*eta_chg > Plim(j)
                        w_stor_k  = -Plim(j)/eta_dc;
                        delta_k   = w_stor(j)+disc - w_stor_k;
                        w_stor(j) = w_stor_k;
                    else
                        delta_k = 0;
                    end
                end
                for i = 1:24
                    soc = soc0 + sum(w_stor(1:i));
                    SOC_re = [SOC_re; soc];
                end
                x49 = max(SOC_re);
                x3(k) = ceil(x49/10000)*10000;   %��������С�߶�10MW
                if x3(k) > X_3
                    X_3 = x3(k);
                end
            end
        end

        % �������仯���ƻس�ŵ繦��
        for i = 1:24
            if w_stor(i) <= 0        %�ŵ磬��Ӧx(i)����0
                x(i) = -w_stor(i)*eta_dc/1;   %w(i)<0�ŵ磬�ų���������1Сʱ*ϵ��0.93���Ƿŵ繦��x(i)
            else                     %��磬w(i) >= 0����Ӧx(i)С��0
                x(i) = -w_stor(i)/1/eta_chg;   %x(i)<0����磬��ص�������w(i),+
            end
        end
        xnew(ii1:ii2) = x;
    end
    
    X1new = xnew;
    X3new = X_3;
end


% Ŀ�꺯��
function Obj = Func_obj(x1, x2, Storage, P_load, Price, Renewable)
    
    % �����������Gridֵ��Grid>0�ӵ�����磬Grid<0
    buy_price   = Price(1, :);   % �������۸񣬵�λ��Ԫ/kWh 
    sell_price  = Price(2, :);   % �����۵�۸񣬵�λ��Ԫ/kWh 
    lease_price = Price(3, 1);   % �����������浥�ۣ�108 Ԫ/kWh
    dr_price    = Price(4, 1);   % ������Ӧ�۸�2 Ԫ/kWh
    model_price = Price(5, 1);   % ����������۸񣬵�λ��Ԫ/kWh
    
    Grid = P_load - (x1 + x2);
    if min(Grid) >= 0   % �ӵ������,�����͵�����
        %����һ��ľ�����
        netgain = 0;
        for i = 1:length(Grid)
            if x1(i) < 0   %Ϊ��������
                if x2(i) <= P_load(i)
                    netgain = netgain + x1(i)*1*buy_price(i);
                else
                    delta(i) = Renewable(i) - x2(i);
                    netgain = netgain - delta(i)*model_price - Grid(i)*buy_price(i);
                end
            else   %Ϊ������ŵ�
                netgain = netgain + x1(i)*1*sell_price(i) ;   % x(i)��+��Ϊ���ܷų������Ĺ��ʣ�*1h*����ۼ�Ϊ��������
            end
        end
                          
        %����һ���������,һ������330�죬ϵͳ�ɱ�590Ԫ/kWh�� 1%������ά�ɱ�,-��������ɱ���+������������-�����ɱ�
        % ����ʱ��-��ͳ����-�����ͣ�ϵͳ�ɱ�590Ԫ/kWh����ά�ɱ�12Ԫ/kWh�������ɱ�380Ԫ/kWh��12�������
        Netgain_total = 330*netgain - 590*Storage/25 - 12*Storage + Storage*lease_price - 380/12*Storage;

        Obj = 25*Netgain_total/(590*Storage/25);     
    else
        Obj = -1000000;   %�������ӻ᷵�͵磬�����潵�����󣬱�֤�������ӽ�����ᱻѡ��
    end  
end






















