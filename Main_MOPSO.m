% ������Ŀ������Ŀ-�����Դ�������ʹ����Ż�ѡ���붯̬����
% ��25��׹��������26��׷�粢������ȫ�������ڷ���
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
n_sence     = 1;      % ���泡��ѭ��������1Ϊ���η�����N_sence=100Ϊ�ೡ������
season      = '��';   % �Ļ򶬼�������
year        = 26      % yearΪ26��27��26��ֻ��475MW�����27���⹲925MW

% ����ΪĬ��ֵ�����޸�
Rate_Load   = 1.5;    %�����������������ֵ
Rate_Power  = 1.0;    %����������
Rate_limt   = [0.95; 1]; %����Դ������Լ����95%-100%
Storage_max = 1500;  %����ܶ������MWh
Storage_min = 1300;  %��С���ܶ������MWh
Max_Iter    = 30;    %����������
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

N_obj     = 1;      %�Ż�Ŀ����
% ����ά�ȣ�ǰ24ά��ʱ���ʣ�25-48���ʵ�ʳ�����49ά��������
if N_obj == 1
    dim = 49;
else
    dim = 50;
    x50_limt   = [10, 18];          %���ܽ���Ľڵ�����
    % �������ܽṹ����������
    % ��һ�У��ڵ�ţ��ڶ��У������й���kW���������У������޹�(kvar)
    Bus    = struct2array(load('Bus_IEEE33.mat'));
    % ��1�д�֧·�ţ���2�д�֧·�׽ڵ㣬��3�д�֧·β�ڵ㣬��4��֧·���裬��5�д�֧·�翹 ��ŷķ��
    Branch = struct2array(load('Branch_IEEE33.mat'));
end

%
sell_price  = ones(1, 24) * 0.5806;   %��λ��0.488 Ԫ/kWh
lease_price = ones(1, 24) * 108;     %�����������浥�ۣ�108 Ԫ/kWh

% *********************���������뼰�䲻ȷ��������*********************** %
% ���װ������������Դ�����������÷�������_2024.10.21��
Capacity_WT  = [0, 45];        %���װ��������45��ǧ�ߣ�26��ײ���
Capacity_PV  = [47.5, 47.5];   %���װ��������47.5��ǧ�ߣ�25��ײ���
if strcmpi(season, '��') == 1
    Source_coef = xlsread('Data_Input.xlsx', '���ϵ��', 'A2:B25');   %�ļ�����ϵ��
else
    Source_coef = xlsread('Data_Input.xlsx', '���ϵ��', 'C2:D25');   %��������ϵ��
end
% ����ϵ�����ݡ��ӳص��������Ŀ15�����ӳ���(������ʢ��������)������
if year>24 && year<28
    WT = 10000*Capacity_WT(year-25).*Source_coef(:,1)';   %����������λ����ǧ��ת��kW
    PV = 10000*Capacity_PV(year-25).*Source_coef(:,2)';   %�����������λ����ǧ��ת��kW
end
Renewable = PV + WT;                                      %����Դ=���+���
% ---------------------���������뼰�䲻ȷ���Խ���----------------------- %

% ***********************�������뼰�䲻ȷ��������************************* %
% ������Դ�����ӳ��ϵ�����룩��ҵ԰Դ���ɴ�һ�廯��Ŀǰ�ڸ��ɵ��鱨�棨10�·ݣ� (11.11-y��v3)��
% ���ݷ��������㸺�ɣ����ճ������������ɵ���=1��1.5Ϊ��׼����
Load_access = ones(1, 1);   %���ɽ���,����8��1��ʾ���룬0��ʾ�����룬
Load_DR     = [0, 0];  %0-10%��������Ӧ�����ɱ�������Ӧʱ��2Сʱ
Uncertainty = '��̬';  %��ȷ���Էֲ�����̬�ֲ�����ȷֲ�

if length(Load_access) == 1
%     Load_Max  = xlsread('Data_Input.xlsx', '���и���', 'B3:C5')./0.85;    %��󸺺�
    %���ݳ����͸�����ȼ��㸺��
    Load_Coef = xlsread('Data_Input.xlsx', '���и���', 'B7:C30');   %����ϵ��
    if strcmpi(season, '��') == 1
        Load_coef = Load_Coef(:, 2:2:end);     %�ļ�����ϵ��
        Load_max  = ones(2, 1) .* sum(Renewable) / sum(Load_coef) /10000 /0.85;
    else
        Load_coef = Load_Coef(:, 1:2:end-1);   %��������ϵ��
        Load_max  = ones(2, 1) .* sum(Renewable) / sum(Load_coef) /10000 /0.85;
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
% Load_growth = mean([Load_plan(2)/Load_plan(1); Load_plan(3)/Load_plan(2)]);   %����������
Load_growth = Load_plan(2)/Load_plan(1);   %����������
if year>24 && year<28
    P_load = 10000*sum(Load_access.* Load_max(year-25, :).*Load_coef, 2)'.*0.85;
else if year>27
        growth = Load_plan(end)*(Load_growth^(year-27));   %��������
        P_load = 10000*sum(Load_access.* Load_max(end, :).*Load_coef, 2)'.*0.85*growth;
    else
        disp('����year�������������24 ');   %��λ����ǧ��ת��kW
        return
    end
end

P_load        = P_load * Rate_Load;   %
Load_rate_cal = sum(P_load) / sum(Renewable)
WT = WT .* Rate_Power;
PV = PV .* Rate_Power;
Renewable = Renewable .* Rate_Power;                       %����������
% -----------------------�������뼰�䲻ȷ���Խ���------------------------- %

SOC_0       = 0.3;   %����ϵͳ��ʼSOC
SOC_limt    = [0.1; 1];     %SOCԼ����10%-100%

%% �����Ż������������
data_price  = xlsread('Data_Input.xlsx', 'Price');    %�۸�
buy_price   = data_price(:,1)';   %��λ��Ԫ/kWh
dr_price    = data_price(:,4)';   %������Ӧ�۸�2 Ԫ/kWh
model_price = 0;                  %����������۸�Ԫ/kWh
Price       = [buy_price; sell_price; lease_price; dr_price; model_price*ones(1,24)];

%% ���Ǹ������صĴ��������Ż�,ʤ��Դ�ɲ�ȷ�����泡��
% ����3������ԭ�����ݷֲ���-+3sigma��Χ�ڵĸ���Ϊ99.73%
% ��˸��ݲ�ȷ����Χ�����������ͳ�Ʒֲ��ı�׼��

N_s    = 1000*ceil(n_sence/100);   %���ɳ�����
N_sence = 100*ceil(n_sence/100);   %������������������100��
% LHS����Դ�ɲ�ȷ���Եķ��泡��
Load_all = [];   %1000*24���ո��ɷ��泡��
EnWT_all = [];   %1000*24���շ��������泡��
EnPV_all = [];   %1000*24���չ���������泡��
Ener_all = [];   %1000*24���շ��������泡��
if strcmpi(Uncertainty, '��̬') == 1
    for t = 1:24
        mu_l(t)     = P_load(t)*1;
        sigma_l(t)  = P_load(t)*Uncertain_load/3;

        mu_wt(t)    = WT(t);
        sigma_wt(t) = WT(t)*Uncertain_source/3;

        mu_pv(t)    = PV(t);
        sigma_pv(t) = PV(t)*Uncertain_source/3;       

        load_temp   = sort(lhsnorm(0, 1, N_s), 'descend');
        Load_temp   = load_temp .* sigma_l(t) + mu_l(t);
        ener_temp   = sort(lhsnorm(0, 1, N_s), 'descend');
        EnWT_temp   = ener_temp .* sigma_wt(t) + mu_wt(t);
        EnPV_temp   = ener_temp .* sigma_pv(t) + mu_pv(t);
        Ener_temp   = EnWT_temp + EnPV_temp;

        Result_mu(t, 1) = mean(Load_temp);
        Result_mu(t, 2) = mean(Ener_temp);
        Result_sigma(t, 1) = std(Load_temp);
        Result_sigma(t, 2) = std(Ener_temp);

        Load_all(:,t) = sort(Load_temp, 'descend');
        EnWT_all(:,t) = sort(EnWT_temp, 'descend');
        EnPV_all(:,t) = sort(EnPV_temp, 'descend');
        Ener_all(:,t) = sort(Ener_temp, 'descend');
    end
else
    for t = 1:24
        load_max = P_load(t)*(1+Uncertain_load);
        load_min = P_load(t)*(1-Uncertain_load);
        
        wt_max   = WT(t)*(1+Uncertain_source);
        wt_min   = WT(t)*(1-Uncertain_source);
        
        pv_max   = PV(t)*(1+Uncertain_source);
        pv_min   = PV(t)*(1-Uncertain_source);
        
        Load_temp = linspace(load_max, load_min, N_s)';
        EnWT_temp = linspace(wt_max, wt_min, N_s)';
        EnPV_temp = linspace(pv_max, pv_min, N_s)';
        Ener_temp = EnWT_temp + EnPV_temp;
        
        Load_all(:,t) = sort(Load_temp, 'descend');
        EnWT_all(:,t) = sort(EnWT_temp, 'descend');
        EnPV_all(:,t) = sort(EnPV_temp, 'descend');
        Ener_all(:,t) = sort(Ener_temp, 'descend');            
    end
end
% �����²������ٳ���������1000������100
Load_All = [];   %100*24��ÿ��Ϊ�ո���ʱ������
EnWT_All = [];   %100*24��ÿ��Ϊ�շ�����ʱ������
EnPV_All = [];   %100*24��ÿ��Ϊ�չ������ʱ������
Ener_All = [];   %100*24��ÿ��Ϊ�ճ���ʱ������
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

figure(102)
plot(Load_All'./1000, '--')
hold on
plot(Ener_All'./1000, '-')
hold on
l2=xlabel('t/h');
set(l2,'Fontname', 'Times New Roman','FontSize',12)
l3=ylabel('P/MW');
set(l3,'Fontname', 'Times New Roman','FontSize',12)
set(gca,'FontName','Times New Roman','FontSize',12)
xlim([0, 24])
close figure 102

%% PSO�㷨����
% ����ά�ȣ�ǰ24ά��ʱ���ʣ�25-48���ʵ�ʳ�����49ά��������
% �˴�����Ϊ���ܷų��Ĺ��ʣ�+�����͸����ܳ��Ĺ��ʣ�-��
% ���ܵ���ڵĵ����仯���ŵ�Ϊ����*1/0.93��+�������Ϊ����*1*0.93��-��

w_max      = 0.9;   %������ϵ��
w_min      = 0.4;   %��С����ϵ��
v_max      = 20000*ones(1,dim);     %�ٶ��Ͻ�
v_max(1, 1:24) = Storage_min*0.1*1000*ones(1,24);     %�ٶ��Ͻ�
v_max(1, 25:48) = max(Renewable)*0.1*ones(1,24);
v_max(49)  = Storage_min*0.5*1000;
if N_obj == 2
    v_max(50) = ceil((x50_limt(2)-x50_limt(1))/5);
end
v_min      = -v_max;

%% ���Ǹ������صĴ��������Ż�������ÿ�������µ����Ŵ��ܣ�
%  ȡ95��λ���Ľ����Ϊ��95%�����Ŷȱ�֤�Ż�����ܸ���Դ�ɵĲ�ȷ����
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
    Price_ns     = Price;   %���ݼ۸��Ƿ񲨶��趨�����ɵ�ǰ�����۸�
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
    if n_sence == 1
        Load_DR_ns = Load_DR(1);
    else
        DR_NS = linspace(Load_DR(1), Load_DR(2), n_sence);
        Load_DR_ns = DR_NS(ns);
    end

   %% PSOѭ��
%     Result = [];   %1-24��Ϊ���Ź������ߣ�25-48��Ϊʵ�ʷ�������49��Ϊ����
%     Grid   = [];   %
    
    for i=1:24
        if sell_price(i) > buy_price(i)
            X_upper(i) = 0 ;           %ǧ�ߣ���ŵ����0.5C���ŵ�Ϊ(+)
            X_lower(i) = -Storage_min/2;   %ǧ�ߣ���ŵ����0.5C�����Ϊ(-)
        else
            X_upper(i) = Storage_min/2 ;   %ǧ�ߣ���ŵ����0.5C���ŵ�Ϊ(+)
            X_lower(i) = -Storage_min/2;   %ǧ�ߣ���ŵ����0.5C�����Ϊ(-)      
        end
    end    
    for i = 25:48
        X_upper(i) = renewable_ns(i-24);
%         X_lower(i) = min(Rate_limt(1)*renewable_ns(i-24), pload_ns(i-24));
        if renewable_ns(i-24) <= pload_ns(i-24)
            X_lower(i) = renewable_ns(i-24);
        else
%             X_lower(i) = min(Rate_limt(1)*renewable_ns(i-24), pload_ns(i-24));
            X_lower(i) = Rate_limt(1)*renewable_ns(i-24);
        end
    end
    load_proportion = sum(pload_ns) / sum(P_load);
    if load_proportion < 0.7
        pop = N_pop*1.5;    %����
    else
        pop = N_pop;
    end
    X_upper(49) = Storage_max;
    X_lower(49) = Storage_min;
    if N_obj == 2
        X_upper(50) = x50_limt(2);
        X_lower(50) = x50_limt(1); 
    end
    

    
    X = initialization(pop, X_upper, X_lower, dim);   %��ʼ����Ⱥλ��
    X(:, 49) = round(X(:, 49)/10000)*10000;           %��������С�߶�10MW
    if N_obj == 2
        X(:, 50) = round(X(:, 50));                   %λ��
    end
    V = initialization(pop, v_max, v_min, dim);       %��ʼ����Ⱥ�ٶ�
%     Plim = pload_ns - X(i,25:48);   %���ܷŵ����ƣ�������������͵�
    for i = 1:pop
        Plim = pload_ns - X(i,25:48);   %���ܷŵ����ƣ�������������͵�
        X(i,:) = ConstraintCheck(X(i,:), X(i, 49), SOC_limt, SOC_0, Plim);
    end

    fx = zeros(N_obj, pop);
    % ��������Ŀ�꺯��1�����ܾ����ԣ�ȡ���
    for i = 1:pop
        Storage = X(i, 49);
        fx(1, i) = fobj(X(i,:), X(i, 49), pload_ns, Price_ns, renewable_ns);   %������Ӧ��ֵ
    end     
    % ��������Ŀ�꺯��2���ڵ��ѹ����������ȡ���
    if N_obj == 2
        V_bus = [];   % 24��
        for k = 1:24
            P_new = [EnWT_All(ns, k), EnPV_All(ns, k)];
            Location_Storage = X(i, 50);   %����λ��
            v_bus = Power_flow(X(i, k), Location_Storage, pload_ns(k), P_new, Bus, Branch);
            V_bus = [V_bus, v_bus];
        end
        for k = 1:size(V_bus, 1)
            vbus = V_bus(k, :);
            temp_fx2(k) = sum(abs(vbus-mean(vbus)));
        end
        fx(2, i)  = 1/sum(temp_fx2);   % 1/ƽ���ڵ��ѹ������
    end
    
    pBest    = X;                  % ����ʼ��Ⱥ��Ϊ��ʷ����
    fx_pBest = fx;                 % ��¼��ʼȫ�����Ž�,Ĭ���Ż����ֵ

    % ��¼��ʼȫ�����Ž�
    [~, index] = max(sum(fx, 1));   %Ѱ����Ӧ������λ��    
%     fx_gBest = fx(:, index);   %��¼��Ӧ��ֵ��λ��
    fx_gBest = -inf;   %��¼��Ӧ��ֵ��λ��
    gBest    = X(index(1),:);
    gBest_index = index(1);

    Xnew     = X;               %��λ��
    fx_New   = fx;   %��λ����Ӧ��ֵ
    
    %% ***********************PSOѭ����ʼ************************* %
    for t = 1:Max_Iter
        wait = ceil(100*ns/n_sence)-1 + ceil(100*t/Max_Iter)/100;
        wait_str = ['���泡��ѭ����� ', num2str(wait), '%'];
        waitbar(wait/100, hwait, wait_str);
        % ������Ⱥ��Ӧ��
        for i = 1:pop
            w  = w_max-(w_max-w_min)*(t)^2/(Max_Iter)^2;   %����Ȩ�ظ���
            c1 = (0.5-2.5)*t/Max_Iter+2.5;                 %��������c1����
            c2 = (2.5-0.5)*t/Max_Iter+0.5;                 %��������c2����
            r1 = rand(1,dim);
            r2 = rand(1,dim);
            V(i,:) = w*V(i,:) + c1.*r1.*(pBest(i,:) - X(i,:)) + c2.*r2.*(gBest - X(i,:));
            V(i,:) = BoundaryCheck(V(i,:),v_max,v_min,dim);   %�ٶȱ߽��鼰Լ��
            Xnew(i,:) = X(i,:) + V(i,:);                      %λ�ø���
            
            % ----------------���ӱ߽��⼰Լ��------------------ %
            if N_obj == 1
                for j = 25:49
                    if Xnew(i,j) > X_upper(j)
                        Xnew(i,j) = X_upper(j);
                    end
                    if Xnew(i,j) < X_lower(j)
                        Xnew(i,j) = X_lower(j);
                    end
                end
            else
                for j = 25:50
                    if Xnew(i,j) > X_upper(j)
                        Xnew(i,j) = X_upper(j);
                    end
                    if Xnew(i,j) < X_lower(j)
                        Xnew(i,j) = X_lower(j);
                    end
                end
                Xnew(:, 50)  = round(Xnew(:, 50));               %����λ��
            end  
            Xnew(:, 49)  = round(Xnew(:, 49)/10000)*10000;   %��������С�߶�10MW
            
            Storage = Xnew(i, 49);
            Plim = pload_ns - Xnew(i,25:48);   %���ܷŵ����ƣ�������������͵�
            Xnew(i,:) = ConstraintCheck(Xnew(i,:), Storage, SOC_limt, SOC_0, Plim);
            
            Grid(:, i) = (pload_ns - (Xnew(i,1:24) + Xnew(i,25:48)))'./1000;
            if min(Grid(:, i)) < 0
                Grid_index(1, i) = -1;
            else
                Grid_index(1, i) = 1;
            end
            
            % ��������Ŀ�꺯��1�����ܾ����ԣ�ȡ���
            for i = 1:pop
                fx_New(1, i) = fobj(X(i,:), Storage, pload_ns, Price_ns, renewable_ns);   %������Ӧ��ֵ
            end
            % ��������Ŀ�꺯��2���ڵ��ѹ����������ȡ���
            if N_obj == 2
                V_bus = [];   % 24��
                for k = 1:24
                    P_new = [EnWT_All(ns, k), EnPV_All(ns, k)];
                    Location_Storage = X(i, 50);   %����λ��
                    v_bus = Power_flow(X(i, k), Location_Storage, pload_ns(k), P_new, Bus, Branch);
                    V_bus = [V_bus, v_bus];
                end
                for k = 1:size(V_bus, 1)
                    vbus = V_bus(k, :);
                    temp_fx2(k) = sum(abs(vbus-mean(vbus)));
                end
                fx_New(2, i)  = 1/sum(temp_fx2);   % 1/ƽ���ڵ��ѹ������
            end
             % ������ʷ����ֵ��Ŀ�꺯����Ȩ��
            if sum(fx_New(:, i)) >= sum(fx_pBest(:, i))
                pBest(i,:)     = Xnew(i,:);
                fx_pBest(:, i) = fx_New(:, i);    
            end
            % ����ȫ������ֵ������Ŀ�꺯����Ȩ��
            if sum(fx_New(:, i)) >= sum(fx_gBest)
                fx_gBest = fx_New(:, i);
                gBest    = Xnew(i,:);
                gBest_index = i;
            end 
        end

        X  = Xnew;
        fx = fx_New;
        % ��¼��ǰ��������ֵ��������Ӧ��ֵ
        Best_Pos(t,:)   = gBest;       %��¼ȫ�����Ž�
        Best_fitness    = fx_gBest;    %��¼���Ž����Ӧ��ֵ
        IterCurve(t, :) = fx_gBest;    %��¼��ǰ���������Ž���Ӧ��ֵ
    end
    %% -----------------------PSOѭ������------------------------- %        
    
    Result = [X'./1000; fx];
    Result_All{1, ns} = Result;
    
    temp_gBest = gBest';
    temp_gBest(1:49) = temp_gBest(1:49)./1000;   %��λkWתMW
    Max_Pdelta = max(renewable_ns - pload_ns)/1000;  %������-��������ֵ��MW
    Max_Pstor = max(abs(temp_gBest(1:24)));          %����ŵ繦�ʣ�MW
    Rate = sum(temp_gBest(25:48))/sum(renewable_ns./1000);  %�����ʣ�%
    Elec_rate = sum(temp_gBest(25:48))/sum(pload_ns/1000);  %����Դ����ռ�ȣ�%
    grid_index = Grid_index(gBest_index);
    Result_Opt(:, ns) = [temp_gBest; ns; Max_Pdelta; Max_Pstor; Rate; Elec_rate; grid_index];
    
%     temp_grid = pload_ns - (gBest(25:48) + gBest(1:24));
%     temp_grid = temp_grid./1000;   %��λkWתMW
%     Grid_Opt(:, ns) = temp_grid'; 
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

Result_sorted = sortrows(Result_Opt', 49);
if ns == 1
    opt_ns = ns;
else
    opt_ns = floor(ns*0.5);
end
Opt_ns = Result_sorted(opt_ns, dim+1);
Grid_opt = Grid_Opt(:, Opt_ns);


%% ������
P_stor_opt  = Result_Opt(1:24, Opt_ns)';     %����24Сʱ���ʣ�kW
disp('���������Ż����������MW/����MWh')
max_Pstor = max(abs(P_stor_opt))
Storage_opt = Result_Opt(49, Opt_ns)
disp('����-����������������������MWh��')
max_Pdelta = max(Ener_All(Opt_ns, :)/1000 - Load_All(Opt_ns, :)/1000);
plim = Load_All(Opt_ns, :) - Ener_All(Opt_ns, :);
Sum_Pchr = -sum(plim(find(plim<0)))/1000;
[max_Pdelta, Sum_Pchr]
disp('�����ʣ�����Դ����ռ�ȣ�%�����õ�����������������������ǧ��ʱ��')
rate      = sum(Result_Opt(25:48, Opt_ns))/sum(Ener_All(Opt_ns, :)/1000);
elec_rate = sum(Result_Opt(25:48, Opt_ns))/sum(Load_All(Opt_ns, :)/1000);
Elec_generation  = sum(Result_Opt(25:48, Opt_ns)*1000)*330/100000000;
Elec_consumption = sum(P_load)*330/100000000;
Elec_buy = Elec_consumption - Elec_generation;
[rate, elec_rate, Elec_consumption, Elec_generation, Elec_buy]

Result = Result_All{1, Opt_ns};


%% �Ż������ͼ
figure(1)
stairs(Load_All(Opt_ns, :)/1000,  'Color', 'b', 'LineWidth', 2)
hold on
stairs(Ener_All(Opt_ns, :)/1000, 'Color', 'g', 'LineWidth', 2)
hold on
line([0, 24], [mean(Load_All(Opt_ns, :))/1000, mean(Load_All(Opt_ns, :))/1000], 'linestyle','--', 'color', 'b')
text(24.1, mean(Load_All(Opt_ns, :)/1000), num2str(mean(Load_All(Opt_ns, :))/1000))
hold on
line([0, 24], [mean(Ener_All(Opt_ns, :))/1000, mean(Ener_All(Opt_ns, :))/1000], 'linestyle','--', 'color', 'g')
text(24.1, mean(Ener_All(Opt_ns, :))/1000, num2str(mean(Ener_All(Opt_ns, :))/1000))
legend('����', '������')
xlim([0, 24])
xlabel('t/h')
ylabel('MW')


figure(2)
% ���������¶�Ӧ���չ�������
stairs(P_stor_opt, '-s','Color',[1 0.64706 0], ...
     'MarkerEdgeColor', 'k','MarkerFaceColor','r','LineWidth', 2)
hold on
line([0,25], [0,0],'linestyle','--','color','r')
xlim([0, 24])
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
x    = P_stor_opt;
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
Storage_dis = [];
Storage_l = Storage_min/1000;
Storage_h = Storage_max/1000;
po_g = find(Result_Opt(dim,:) > Storage_l & Result_Opt(dim,:) < Storage_h);
Storage_slected = Result_Opt(dim,po_g);
for i = 1:((Storage_h-Storage_l)/10-1)
    storage_temp     = i*10 + Storage_l;
    Storage_dis(1,i) = storage_temp;
    Storage_dis(2,i) = length(find(Storage_slected == storage_temp));
end
bar(Storage_dis(1,:), Storage_dis(2,:), 'b')
storage_opt = Storage_dis(1, find(Storage_dis(2,:) == max(Storage_dis(2,:))));
num_opt     = Storage_dis(2, find(Storage_dis(2,:) == max(Storage_dis(2,:))));
xlabel('MW')
ylabel('Ƶ��')
title('Ƶ���ֲ�ֱ��ͼ')

figure(6)
stairs(Grid_opt, '-s','Color',[1 0.64706 0], ...
     'MarkerEdgeColor', 'k','MarkerFaceColor','r','LineWidth', 2)
hold on
line([0,25], [0,0],'linestyle','--','color','r')
xlim([0, 24])
xlabel('t/h')
ylabel('����/MW')
title('��������')


%% ����Ϊ����

%% ����Ⱥ��ʼ������,��ʼ��λ�û��ٶ�
function X = initialization(pop, ub, lb, dim)
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
function [X] = BoundaryCheck(x, ub, lb, dim)
    % dimΪ���ݵ�ά�ȴ�С
    % xΪ�������ݣ�ά��Ϊ[1,dim];
    % ubΪ�����ϱ߽磬ά��Ϊ[1,dim]
    % lbΪ�����±߽磬ά��Ϊ[1,dim]
    for i = 1:dim
        if x(i)>ub(i)
           x(i) = ub(i); 
        end
        if x(i)<lb(i)
            x(i) = lb(i);
        end
    end
    X = x;
end

%% ���ܳ�ŵ繦��Լ������
function Xnew = ConstraintCheck(X, Storage, SOC_limt, SOC_0, Plim)
    % dimΪ���ݵ�ά�ȴ�С
    % xΪ�������ݣ�ά��Ϊ[1,dim];
    % SOC_limt��SOC���½�Լ����[0.1, 1]
    % Rate_limt�����������½�Լ����[0.95, 1]
    x       = X(1:24);   %����ʱ����
    SOC     = [];
    SOC_re  = [];
    SOC_min = Storage*SOC_limt(1);
    SOC_max = Storage*SOC_limt(2);
    eta_chg = 0.927;   % ���Ч��
    eta_dc  = 0.928;   % �ŵ�Ч��
    
    % ��ŵ�Լ��˼·������ŵ繦��ת������صĵ����仯w�����ӻ���٣�����������Ϊ0��
    % w���ۼӼ���SOC��w������Լ����ŵ�ʱ�̡�
    
    % Լ��ÿ�ճ�ŵ������
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
%         po1 = find(po_char < po_lim(1));
%         po2 = po_char(po1);                %����ǰ�ĳ��ʱ��
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
                k = po4(i);
                if -(w_stor(k)+disc+delta_k)*eta_chg > Plim(k)
                    w_stor_k  = -Plim(k)/eta_dc;
                    delta_k   = w_stor(k)+disc - w_stor_k;
                    w_stor(k) = w_stor_k;
                else
                    delta_k = 0;
                end
            end
            for i = 1:24
                soc = soc0 + sum(w_stor(1:i));
                SOC_re = [SOC_re; soc];
            end
            X49 = max(SOC_re);
            X(49) = ceil(X49/10000)*10000;   %��������С�߶�10MW
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
    
    X(1:24) = x;   
    Xnew    = X;
end

%% Ŀ�꺯��,�޸�obj1��obj2����
% ��С����200MW/400MWh
function Obj = fobj(x, Storage, P_load, Price, Renewable)
    %�����������Gridֵ��Grid>0�ӵ�����磬Grid<0
    buy_price   = Price(1, :);   % �������۸񣬵�λ��Ԫ/kWh 
    sell_price  = Price(2, :);   % �����۵�۸񣬵�λ��Ԫ/kWh 
    lease_price = Price(3, 1);   % �����������浥�ۣ�108 Ԫ/kWh
    dr_price    = Price(4, 1);   % ������Ӧ�۸�2 Ԫ/kWh
    model_price = Price(5, 1);   % ����������۸񣬵�λ��Ԫ/kWh
    
    Grid = P_load - (x(25:48) + x(1:24));
    if min(Grid) >= 0   % �ӵ������,�����͵�����
        %����һ��ľ�����
        netgain = 0;
        for i = 1:24
            if x(i) < 0   %Ϊ��������
                if x(i+24) <= P_load(i)
                    netgain = netgain + x(i)*1*buy_price(i);
                else
                    delta(i) = Renewable(i) - x(i+24);
                    netgain = netgain - delta(i)*model_price - Grid(i)*buy_price(i);
                end
            else  %Ϊ������ŵ�
                netgain = netgain + x(i)*1*sell_price(i) ;   % x(i)��+��Ϊ���ܷų������Ĺ��ʣ�*1h*����ۼ�Ϊ��������
            end
        end
                          
        % ����һ���������,һ������330�죬ϵͳ�ɱ�880Ԫ/kWh�� 1%������ά�ɱ���-��������ɱ���+������������
        % ����ʱ��-��ͳ����-�����ͣ�ϵͳ�ɱ�590Ԫ/kWh����ά�ɱ�12Ԫ/kWh�������ɱ�380Ԫ/kWh��12�������
        Netgain_total = 330*netgain - 590*Storage/25 - 12*Storage + Storage*lease_price - 0*Storage;   %26��
%         Netgain_total = 330*netgain - 590*Storage/25 - 12*Storage + Storage*lease_price - 380/12*Storage;

        Obj = Netgain_total/(590*Storage/25);     
    else
        Obj = -10000;   %�������ӻ᷵�͵磬�����潵�����󣬱�֤�������ӽ�����ᱻѡ��
    end     
end






















