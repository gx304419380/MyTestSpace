% 大唐项目储能项目-清洁能源基地新型储能优化选型与动态配置
% 从25年底光伏并网，26年底风电并网，做全生命周期分析
% 从负荷、源侧和商业模式三个角度，多因素分析
% 单层PSO算法,储能功率修正后调整容量，
% 2025.01.22
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
clc

tic;
Font_size = 12;   %字体大小
format short;
warning off
set(0, 'defaultAxesFontName', 'Monospaced');
hwait = waitbar(0,'数据输入，请等待>>>>>>>>');
%% 用户输入
n_sence     = 1;      % 仿真场景循环次数，1为单次分析，N_sence=100为多场景分析
season      = '夏';   % 夏或冬季典型日
year        = 26      % year为26、27，26年只有475MW光伏，27年风光共925MW

% 以下为默认值，可修改
Rate_Load   = 1.5;    %负荷运行与风光出力比值
Rate_Power  = 1.0;    %风光出力倍率
Rate_limt   = [0.95; 1]; %新能源消纳率约束，95%-100%
Storage_max = 1500;  %最大储能额定容量，MWh
Storage_min = 1300;  %最小储能额定容量，MWh
Max_Iter    = 30;    %最大迭代次数
N_pop       = 100;   %种群粒子数量
%

if n_sence == 1
    Price_change = [0, 0, 0, 0, 0];   %标记价格变化，0不变，1变化,[0, 1, 1, 0, 0]
    % 价格变化范围倍数，第一行上限，第二行下限，用于生成价格的不确定范围
    Price_range = [1, 1; 1.2049, 1; 1.1852, 0.8148; 1, 1; 1, 1]';  
    Price_dist  = 1;   % 价格分布，1为随机生成，2为均匀分布
    % 风光出力不确定性，参数默认为0.1时，表示出力在基准值的90%-110%范围
    Uncertain_source = 0.0;
    % 负荷不确定性，参数默认为0.5时，表示负荷在基准值的100%-200%范围
    Uncertain_load   = 0.0; 
else
    Price_change = [0, 1, 1, 0, 0];   %标记价格变化，0不变，1变化,[0, 1, 1, 0, 0]
    % 价格变化范围倍数，第一行上限，第二行下限，用于生成价格的不确定范围
    Price_range = [1, 1; 1.2049, 1; 1.1852, 0.8148; 1, 1; 1, 1]';  
    Price_dist  = 1;   % 价格分布，1为随机生成，2为均匀分布
    % 风光出力不确定性，参数默认为0.1时，表示出力在基准值的90%-110%范围
    Uncertain_source = 0.1;   % 默认0.1，用户可修改
    % 负荷不确定性，参数默认为0.5时，表示负荷在基准值的100%-200%范围
    Uncertain_load   = 0.5;   % 默认0.5，用户可修改
end

N_obj     = 1;      %优化目标数
% 粒子维度，前24维日时序功率，25-48风光实际出力，49维储能容量
if N_obj == 1
    dim = 49;
else
    dim = 50;
    x50_limt   = [10, 18];          %储能接入的节点限制
    % 导入网架结构，电力潮流
    % 第一列：节点号，第二列：负荷有功（kW），第三列：负荷无功(kvar)
    Bus    = struct2array(load('Bus_IEEE33.mat'));
    % 第1列存支路号，第2列存支路首节点，第3列存支路尾节点，第4存支路电阻，第5列存支路电抗 （欧姆）
    Branch = struct2array(load('Branch_IEEE33.mat'));
end

%
sell_price  = ones(1, 24) * 0.5806;   %单位：0.488 元/kWh
lease_price = ones(1, 24) * 108;     %容量租赁收益单价，108 元/kWh

% *********************风光出力输入及其不确定性输入*********************** %
% 风光装机容量数据来源：《储能配置方案报告_2024.10.21》
Capacity_WT  = [0, 45];        %风电装机容量，45万千瓦，26年底并网
Capacity_PV  = [47.5, 47.5];   %光伏装机容量，47.5万千瓦，25年底并网
if strcmpi(season, '夏') == 1
    Source_coef = xlsread('Data_Input.xlsx', '风光系数', 'A2:B25');   %夏季出力系数
else
    Source_coef = xlsread('Data_Input.xlsx', '风光系数', 'C2:D25');   %冬季出力系数
end
% 出力系数根据《河池地区光伏项目15分钟钟出力(常吉、盛景、拉才)》计算
if year>24 && year<28
    WT = 10000*Capacity_WT(year-25).*Source_coef(:,1)';   %风电出力，单位从万千瓦转成kW
    PV = 10000*Capacity_PV(year-25).*Source_coef(:,2)';   %光伏出力，单位从万千瓦转成kW
end
Renewable = PV + WT;                                      %新能源=光伏+风电
% ---------------------风光出力输入及其不确定性结束----------------------- %

% ***********************负荷输入及其不确定性输入************************* %
% 数据来源：《河池南丹（天峨）工业园源网荷储一体化项目前期负荷调查报告（10月份） (11.11-y改v3)》
% 根据风光出力计算负荷，按照出力电量：负荷电量=1：1.5为基准计算
Load_access = ones(1, 1);   %负荷接入,数量8，1表示接入，0表示不接入，
Load_DR     = [0, 0];  %0-10%，需求响应减负荷比例，响应时间2小时
Uncertainty = '正态';  %不确定性分布，正态分布或均匀分布

if length(Load_access) == 1
%     Load_Max  = xlsread('Data_Input.xlsx', '所有负荷', 'B3:C5')./0.85;    %最大负荷
    %根据出力和负荷配比计算负荷
    Load_Coef = xlsread('Data_Input.xlsx', '所有负荷', 'B7:C30');   %负荷系数
    if strcmpi(season, '夏') == 1
        Load_coef = Load_Coef(:, 2:2:end);     %夏季负荷系数
        Load_max  = ones(2, 1) .* sum(Renewable) / sum(Load_coef) /10000 /0.85;
    else
        Load_coef = Load_Coef(:, 1:2:end-1);   %冬季负荷系数
        Load_max  = ones(2, 1) .* sum(Renewable) / sum(Load_coef) /10000 /0.85;
    end   
else
    Load_max  = xlsread('Data_Input.xlsx', '负荷', 'B4:Q5');    %最大负荷
    Load_Coef = xlsread('Data_Input.xlsx', '负荷', 'B7:Q30');   %负荷系数
    if strcmpi(season, '夏') == 1
        Load_max  = Load_max(:, 2:2:end);      %夏季最大负荷
        Load_coef = Load_Coef(:, 2:2:end);     %夏季负荷系数
    else
        Load_max  = Load_max(:, 1:2:end-1);    %冬季最大负荷
        Load_coef = Load_Coef(:, 1:2:end-1);   %冬季负荷系数
    end
end
Load_Max  = sum(Load_max, 2);
Load_plan = sum(Load_access.*Load_max, 2).*0.85;   %25年以后的负荷规划值.0.85为同时率
% Load_growth = mean([Load_plan(2)/Load_plan(1); Load_plan(3)/Load_plan(2)]);   %负荷增长率
Load_growth = Load_plan(2)/Load_plan(1);   %负荷增长率
if year>24 && year<28
    P_load = 10000*sum(Load_access.* Load_max(year-25, :).*Load_coef, 2)'.*0.85;
else if year>27
        growth = Load_plan(end)*(Load_growth^(year-27));   %负荷增长
        P_load = 10000*sum(Load_access.* Load_max(end, :).*Load_coef, 2)'.*0.85*growth;
    else
        disp('错误：year输入有误，需大于24 ');   %单位从万千瓦转成kW
        return
    end
end

P_load        = P_load * Rate_Load;   %
Load_rate_cal = sum(P_load) / sum(Renewable)
WT = WT .* Rate_Power;
PV = PV .* Rate_Power;
Renewable = Renewable .* Rate_Power;                       %风光出力倍率
% -----------------------负荷输入及其不确定性结束------------------------- %

SOC_0       = 0.3;   %储能系统初始SOC
SOC_limt    = [0.1; 1];     %SOC约束，10%-100%

%% 容量优化程序输入参数
data_price  = xlsread('Data_Input.xlsx', 'Price');    %价格
buy_price   = data_price(:,1)';   %单位：元/kWh
dr_price    = data_price(:,4)';   %需求响应价格，2 元/kWh
model_price = 0;                  %弃风弃光充电价格，元/kWh
Price       = [buy_price; sell_price; lease_price; dr_price; model_price*ones(1,24)];

%% 考虑概率因素的储能配置优化,胜场源荷不确定仿真场景
% 根据3西格玛原则，数据分布在-+3sigma范围内的概率为99.73%
% 因此根据不确定范围计算后续数据统计分布的标准差

N_s    = 1000*ceil(n_sence/100);   %生成场景数
N_sence = 100*ceil(n_sence/100);   %缩减场景数，缩减至100个
% LHS生成源荷不确定性的仿真场景
Load_all = [];   %1000*24，日负荷仿真场景
EnWT_all = [];   %1000*24，日风电出力仿真场景
EnPV_all = [];   %1000*24，日光伏出力仿真场景
Ener_all = [];   %1000*24，日风光出力仿真场景
if strcmpi(Uncertainty, '正态') == 1
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
% 采用下采样减少场景数，从1000缩减到100
Load_All = [];   %100*24，每行为日负荷时间序列
EnWT_All = [];   %100*24，每行为日风电出力时间序列
EnPV_All = [];   %100*24，每行为日光伏出力时间序列
Ener_All = [];   %100*24，每行为日出力时间序列
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

%% PSO算法参数
% 粒子维度，前24维日时序功率，25-48风光实际出力，49维储能容量
% 此处功率为储能放出的功率（+），和给储能充电的功率（-）
% 储能电池内的电量变化，放电为功率*1/0.93（+），充电为功率*1*0.93（-）

w_max      = 0.9;   %最大惯性系数
w_min      = 0.4;   %最小惯性系数
v_max      = 20000*ones(1,dim);     %速度上界
v_max(1, 1:24) = Storage_min*0.1*1000*ones(1,24);     %速度上界
v_max(1, 25:48) = max(Renewable)*0.1*ones(1,24);
v_max(49)  = Storage_min*0.5*1000;
if N_obj == 2
    v_max(50) = ceil((x50_limt(2)-x50_limt(1))/5);
end
v_min      = -v_max;

%% 考虑概率因素的储能配置优化，计算每个场景下的最优储能；
%  取95分位数的结果，为以95%的置信度保证优化结果能覆盖源荷的不确定性
Result_All = {};
Grid_All   = {};
Result_Opt = [];
Grid_Opt   = [];
Fitness    = [];

Storage_max = Storage_max*1000;   %单位从MWh转到kWh
Storage_min = Storage_min*1000;   %单位从MWh转到kWh
for ns = 1:n_sence   % ns = 1:n_sence
    renewable_ns = Ener_All(ns, :);   %当前场景风光出力
    pload_ns     = Load_All(ns, :);   %当前场景负荷
    Price_ns     = Price;   %根据价格是否波动设定，生成当前场景价格
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

   %% PSO循环
%     Result = [];   %1-24行为最优功率曲线，25-48行为实际风光出力，49行为容量
%     Grid   = [];   %
    
    for i=1:24
        if sell_price(i) > buy_price(i)
            X_upper(i) = 0 ;           %千瓦，充放电参数0.5C，放电为(+)
            X_lower(i) = -Storage_min/2;   %千瓦，充放电参数0.5C，充电为(-)
        else
            X_upper(i) = Storage_min/2 ;   %千瓦，充放电参数0.5C，放电为(+)
            X_lower(i) = -Storage_min/2;   %千瓦，充放电参数0.5C，充电为(-)      
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
        pop = N_pop*1.5;    %负荷
    else
        pop = N_pop;
    end
    X_upper(49) = Storage_max;
    X_lower(49) = Storage_min;
    if N_obj == 2
        X_upper(50) = x50_limt(2);
        X_lower(50) = x50_limt(1); 
    end
    

    
    X = initialization(pop, X_upper, X_lower, dim);   %初始化种群位置
    X(:, 49) = round(X(:, 49)/10000)*10000;           %容量，最小尺度10MW
    if N_obj == 2
        X(:, 50) = round(X(:, 50));                   %位置
    end
    V = initialization(pop, v_max, v_min, dim);       %初始化种群速度
%     Plim = pload_ns - X(i,25:48);   %储能放电限制，避免向电网返送电
    for i = 1:pop
        Plim = pload_ns - X(i,25:48);   %储能放电限制，避免向电网返送电
        X(i,:) = ConstraintCheck(X(i,:), X(i, 49), SOC_limt, SOC_0, Plim);
    end

    fx = zeros(N_obj, pop);
    % 计算粒子目标函数1，储能经济性，取最大
    for i = 1:pop
        Storage = X(i, 49);
        fx(1, i) = fobj(X(i,:), X(i, 49), pload_ns, Price_ns, renewable_ns);   %计算适应度值
    end     
    % 计算粒子目标函数2，节点电压波动倒数，取最大
    if N_obj == 2
        V_bus = [];   % 24列
        for k = 1:24
            P_new = [EnWT_All(ns, k), EnPV_All(ns, k)];
            Location_Storage = X(i, 50);   %储能位置
            v_bus = Power_flow(X(i, k), Location_Storage, pload_ns(k), P_new, Bus, Branch);
            V_bus = [V_bus, v_bus];
        end
        for k = 1:size(V_bus, 1)
            vbus = V_bus(k, :);
            temp_fx2(k) = sum(abs(vbus-mean(vbus)));
        end
        fx(2, i)  = 1/sum(temp_fx2);   % 1/平均节点电压波动率
    end
    
    pBest    = X;                  % 将初始种群作为历史最优
    fx_pBest = fx;                 % 记录初始全局最优解,默认优化最大值

    % 记录初始全局最优解
    [~, index] = max(sum(fx, 1));   %寻找适应度最大的位置    
%     fx_gBest = fx(:, index);   %记录适应度值和位置
    fx_gBest = -inf;   %记录适应度值和位置
    gBest    = X(index(1),:);
    gBest_index = index(1);

    Xnew     = X;               %新位置
    fx_New   = fx;   %新位置适应度值
    
    %% ***********************PSO循环开始************************* %
    for t = 1:Max_Iter
        wait = ceil(100*ns/n_sence)-1 + ceil(100*t/Max_Iter)/100;
        wait_str = ['仿真场景循环完成 ', num2str(wait), '%'];
        waitbar(wait/100, hwait, wait_str);
        % 计算种群适应度
        for i = 1:pop
            w  = w_max-(w_max-w_min)*(t)^2/(Max_Iter)^2;   %惯性权重更新
            c1 = (0.5-2.5)*t/Max_Iter+2.5;                 %加速因子c1更新
            c2 = (2.5-0.5)*t/Max_Iter+0.5;                 %加速因子c2更新
            r1 = rand(1,dim);
            r2 = rand(1,dim);
            V(i,:) = w*V(i,:) + c1.*r1.*(pBest(i,:) - X(i,:)) + c2.*r2.*(gBest - X(i,:));
            V(i,:) = BoundaryCheck(V(i,:),v_max,v_min,dim);   %速度边界检查及约束
            Xnew(i,:) = X(i,:) + V(i,:);                      %位置更新
            
            % ----------------粒子边界检测及约束------------------ %
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
                Xnew(:, 50)  = round(Xnew(:, 50));               %储能位置
            end  
            Xnew(:, 49)  = round(Xnew(:, 49)/10000)*10000;   %容量，最小尺度10MW
            
            Storage = Xnew(i, 49);
            Plim = pload_ns - Xnew(i,25:48);   %储能放电限制，避免向电网返送电
            Xnew(i,:) = ConstraintCheck(Xnew(i,:), Storage, SOC_limt, SOC_0, Plim);
            
            Grid(:, i) = (pload_ns - (Xnew(i,1:24) + Xnew(i,25:48)))'./1000;
            if min(Grid(:, i)) < 0
                Grid_index(1, i) = -1;
            else
                Grid_index(1, i) = 1;
            end
            
            % 计算粒子目标函数1，储能经济性，取最大
            for i = 1:pop
                fx_New(1, i) = fobj(X(i,:), Storage, pload_ns, Price_ns, renewable_ns);   %计算适应度值
            end
            % 计算粒子目标函数2，节点电压波动倒数，取最大
            if N_obj == 2
                V_bus = [];   % 24列
                for k = 1:24
                    P_new = [EnWT_All(ns, k), EnPV_All(ns, k)];
                    Location_Storage = X(i, 50);   %储能位置
                    v_bus = Power_flow(X(i, k), Location_Storage, pload_ns(k), P_new, Bus, Branch);
                    V_bus = [V_bus, v_bus];
                end
                for k = 1:size(V_bus, 1)
                    vbus = V_bus(k, :);
                    temp_fx2(k) = sum(abs(vbus-mean(vbus)));
                end
                fx_New(2, i)  = 1/sum(temp_fx2);   % 1/平均节点电压波动率
            end
             % 更新历史最优值，目标函数等权重
            if sum(fx_New(:, i)) >= sum(fx_pBest(:, i))
                pBest(i,:)     = Xnew(i,:);
                fx_pBest(:, i) = fx_New(:, i);    
            end
            % 更新全局最优值，两个目标函数等权重
            if sum(fx_New(:, i)) >= sum(fx_gBest)
                fx_gBest = fx_New(:, i);
                gBest    = Xnew(i,:);
                gBest_index = i;
            end 
        end

        X  = Xnew;
        fx = fx_New;
        % 记录当前迭代最优值和最优适应度值
        Best_Pos(t,:)   = gBest;       %记录全局最优解
        Best_fitness    = fx_gBest;    %记录最优解的适应度值
        IterCurve(t, :) = fx_gBest;    %记录当前迭代的最优解适应度值
    end
    %% -----------------------PSO循环结束------------------------- %        
    
    Result = [X'./1000; fx];
    Result_All{1, ns} = Result;
    
    temp_gBest = gBest';
    temp_gBest(1:49) = temp_gBest(1:49)./1000;   %单位kW转MW
    Max_Pdelta = max(renewable_ns - pload_ns)/1000;  %风光出力-负荷最大差值，MW
    Max_Pstor = max(abs(temp_gBest(1:24)));          %最大充放电功率，MW
    Rate = sum(temp_gBest(25:48))/sum(renewable_ns./1000);  %消纳率，%
    Elec_rate = sum(temp_gBest(25:48))/sum(pload_ns/1000);  %新能源消纳占比，%
    grid_index = Grid_index(gBest_index);
    Result_Opt(:, ns) = [temp_gBest; ns; Max_Pdelta; Max_Pstor; Rate; Elec_rate; grid_index];
    
%     temp_grid = pload_ns - (gBest(25:48) + gBest(1:24));
%     temp_grid = temp_grid./1000;   %单位kW转MW
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


%% 输出结果
P_stor_opt  = Result_Opt(1:24, Opt_ns)';     %储能24小时功率，kW
disp('储能容量优化结果：功率MW/容量MWh')
max_Pstor = max(abs(P_stor_opt))
Storage_opt = Result_Opt(49, Opt_ns)
disp('出力-负荷最大差，风光出力需存容量（MWh）')
max_Pdelta = max(Ener_All(Opt_ns, :)/1000 - Load_All(Opt_ns, :)/1000);
plim = Load_All(Opt_ns, :) - Ener_All(Opt_ns, :);
Sum_Pchr = -sum(plim(find(plim<0)))/1000;
[max_Pdelta, Sum_Pchr]
disp('消纳率，新能源消纳占比（%）；用电量，发电量，购电量（亿千瓦时）')
rate      = sum(Result_Opt(25:48, Opt_ns))/sum(Ener_All(Opt_ns, :)/1000);
elec_rate = sum(Result_Opt(25:48, Opt_ns))/sum(Load_All(Opt_ns, :)/1000);
Elec_generation  = sum(Result_Opt(25:48, Opt_ns)*1000)*330/100000000;
Elec_consumption = sum(P_load)*330/100000000;
Elec_buy = Elec_consumption - Elec_generation;
[rate, elec_rate, Elec_consumption, Elec_generation, Elec_buy]

Result = Result_All{1, Opt_ns};


%% 优化结果画图
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
legend('负荷', '风光出力')
xlim([0, 24])
xlabel('t/h')
ylabel('MW')


figure(2)
% 最优容量下对应的日功率曲线
stairs(P_stor_opt, '-s','Color',[1 0.64706 0], ...
     'MarkerEdgeColor', 'k','MarkerFaceColor','r','LineWidth', 2)
hold on
line([0,25], [0,0],'linestyle','--','color','r')
xlim([0, 24])
xlabel('t/h')
ylabel('功率/MW')
title('储能充放电功率')

figure(3)
if N_obj == 1
    fitness = Fitness(:, opt_ns)';
    t = 1:length(fitness);
    fitness(find(fitness<-1000)) = 0;
    plot(t, fitness)
    % ylim([-3, 3])
    xlabel('迭代次数')
    ylabel('优化目标')
else if N_obj == 2
        subplot(211)
        fitness = Fitness(1:Max_Iter, opt_ns)';
        t = 1:length(fitness);
        fitness(find(fitness<-1000)) = 0;
        plot(t, fitness)
        % ylim([-3, 3])
        xlabel('迭代次数')
        ylabel('优化目标1')
        
        subplot(212)
        fitness = Fitness(Max_Iter+1:Max_Iter*2, opt_ns)';
        t = 1:length(fitness);
        fitness(find(fitness<-1000)) = 0;
        plot(t, fitness)
        % ylim([-3, 3])
        xlabel('迭代次数')
        ylabel('优化目标2')
    end
end


figure(4)
x    = P_stor_opt;
soc0 = Storage_opt*SOC_0;
SOC  = [soc0];
for i = 1:24
    if x(i) > 0        %放电
        w_stor(i) = -x(i)*1/0.93;   %x(i)>0，放电，电池电量减少w(i),-
    else 
        w_stor(i) = -x(i)*1*0.93;   %x(i)<0，充电，电池电量增加w(i),+
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
ylabel('频数')
title('频数分布直方图')

figure(6)
stairs(Grid_opt, '-s','Color',[1 0.64706 0], ...
     'MarkerEdgeColor', 'k','MarkerFaceColor','r','LineWidth', 2)
hold on
line([0,25], [0,0],'linestyle','--','color','r')
xlim([0, 24])
xlabel('t/h')
ylabel('功率/MW')
title('电网购电')


%% 后续为函数

%% 粒子群初始化函数,初始化位置或速度
function X = initialization(pop, ub, lb, dim)
    % pop:为种群数量
    % dim:每个粒子群的维度
    % ub: 为每个维度的变量上边界，维度为[1,dim];
    % lb: 为每个维度的变量下边界，维度为[1,dim];
    % X:  为输出的种群，维度[pop,dim];
    X = zeros(pop, dim); %为X事先分配空间
    for i = 1:pop
       for j = 1:dim
           X(i,j) = (ub(j) - lb(j))*rand() + lb(j);  %生成[lb,ub]之间的随机数
       end
    end
end


%% 边界检查函数
function [X] = BoundaryCheck(x, ub, lb, dim)
    % dim为数据的维度大小
    % x为输入数据，维度为[1,dim];
    % ub为数据上边界，维度为[1,dim]
    % lb为数据下边界，维度为[1,dim]
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

%% 储能充放电功率约束条件
function Xnew = ConstraintCheck(X, Storage, SOC_limt, SOC_0, Plim)
    % dim为数据的维度大小
    % x为输入数据，维度为[1,dim];
    % SOC_limt：SOC上下界约束，[0.1, 1]
    % Rate_limt：消纳率上下界约束，[0.95, 1]
    x       = X(1:24);   %修正时序功率
    SOC     = [];
    SOC_re  = [];
    SOC_min = Storage*SOC_limt(1);
    SOC_max = Storage*SOC_limt(2);
    eta_chg = 0.927;   % 充电效率
    eta_dc  = 0.928;   % 放电效率
    
    % 充放电约束思路：将充放电功率转换到电池的电量变化w（增加或减少）；日增减和为0；
    % w的累加计算SOC，w的正负约束充放电时刻。
    
    % 约束每日充放电量相等
    for i = 1:24
        if x(i) >= 0        %放电
            w_stor(i) = -x(i)*1/eta_dc;   %x(i)>0，放电，电池电量减少w(i)<0
        else if x(i) < 0   %充电
            w_stor(i) = -x(i)*1*eta_chg;   %x(i)<0，充电，电池电量增加w(i)>0
            end
        end
    end
    w_stor = w_stor - (sum(w_stor)-0)/24;
    
    % buy_price<sell_price不能卖电,[1,2,3,4,5,6,7,24]时刻不能买电，x需要小于0
    for i= 1:7
        if w_stor(i) <= 0   %放电，电量减少，w(i)<0
            temp1 = Storage*0.01*rand()*eta_dc;
            w_stor(8:23) = w_stor(8:23) - (temp1 - w_stor(i))/16;
            w_stor(i) = temp1;
        end
    end
    if w_stor(24) <= 0     %放电，电量减少，w(i)<0
        temp2 = Storage*0.01*rand()*eta_dc;
        w_stor(8:23) = w_stor(8:23) - (temp2 - w_stor(i))/16;
        w_stor(24) = temp2;
    end
    
    % 充放电速率约束，不能超过Storage*0.5
    pro_x = zeros(24, 1);     %电量变化w和充放电功率约束下的电量变化比值，大于1则速率超过约束
    for i = 1:24
        if w_stor(i) <= 0   %放电，电量减少
            pro_x(i) = w_stor(i)/(-Storage*0.5*1/eta_dc);
        else if w_stor(i) > 0   %充电，电量增加
                pro_x(i) = w_stor(i)/(Storage*0.5*1*eta_chg);
            end
        end
    end
    if max(pro_x) > 1   %大于1，则充放电功率有超过约束的时刻
        w_stor = w_stor .* (1/max(pro_x));
    end
        
    % SOC约束，min<SOC（t）<max，t\in[1:24]
    soc0   = Storage*SOC_0;
    for i = 1:24
        soc = soc0 + sum(w_stor(1:i));
        SOC = [SOC; soc];
    end
    soc_pro1 = (Storage*(1-SOC_0))/(max(SOC)-soc0);
    soc_pro2 = (Storage*(SOC_0-0.1))/(soc0-min(SOC)); 
    w_stor = w_stor .* min(soc_pro1, soc_pro2);
    
    % 防止返送电约束，负荷-新能源为限制功率，储能放电大于限制，则会出现返送电
    po_lim  = [];
    po_char = find(w_stor > 0);   %充电时刻
    po_disc = find(w_stor <= 0);  %放电时刻
    w_sum   = 0;
    for i = 1:24
        % 发电大于负荷&储能在放电，会返送电，变为充电
        if Plim(i)<0 & w_stor(i)<0
            po_lim = [po_lim; i];                  %返送电时刻
            delta_w = -Plim(i)*eta_dc - w_stor(i);   %容量增加量>0
            w_sum = w_sum + abs(delta_w);             %放电转充电，容量增加
            w_stor(i) = -Plim(i)*eta_chg;             %多发的电都用来充电
        % 发电大于负荷&储能在充电，但充电功率不足，会返送电，增加充电功率
        else if Plim(i)<0 & -w_stor(i)/eta_chg > Plim(i)
                po_lim = [po_lim; i];                  %返送电时刻
                delta_w = -Plim(i)*eta_chg - w_stor(i);   %容量增加量>0
                w_sum = w_sum + abs(delta_w);             %充电变大，容量增加
                w_stor(i) = -Plim(i)*eta_chg;             %多发的电都用来充电
            % 发电小于负荷&储能在放电，但放电功率过大，会返送电，减少放电功率
            else if Plim(i)>0 & -w_stor(i)*eta_dc > Plim(i)
                    po_lim = [po_lim; i];                  %返送电时刻
                    delta_w = -Plim(i)/eta_dc - w_stor(i);   %容量增加量>0
                    w_sum = w_sum + abs(delta_w);             %放电减小，容量增加
                    w_stor(i) = -Plim(i)/eta_dc;             %减少放电
                end
            end
        end
    end
    % 修正返送电时刻功率，导致返送电时刻容量增加，其他时刻需要减少充电量
    if length(po_lim) > 0
%         po1 = find(po_char < po_lim(1));
%         po2 = po_char(po1);                %超限前的充电时刻
        po2 = [1:7];
        char_sum = sum(w_stor(po2));       %超限前充电量，正数
        delta = abs(char_sum) - abs(w_sum);
        if delta > 0
            pro_lim = delta / abs(char_sum);
            w_stor(po2) = w_stor(po2).*pro_lim;
        else
            w_stor(po2) = 0;    %
            po3 = [8:23];
            po4 = setdiff(po3, po_lim);
            disc = delta / length(po4);  %20250108修改，避免修正后放电过大，导致返送电
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
            X(49) = ceil(X49/10000)*10000;   %容量，最小尺度10MW
        end
    end
    
    % 将电量变化反推回充放电功率
    for i = 1:24
        if w_stor(i) <= 0        %放电，对应x(i)大于0
            x(i) = -w_stor(i)*eta_dc/1;   %w(i)<0放电，放出电量除以1小时*系数0.93才是放电功率x(i)
        else                     %充电，w(i) >= 0，对应x(i)小于0
            x(i) = -w_stor(i)/1/eta_chg;   %x(i)<0，充电，电池电量增加w(i),+
        end
    end
    
    X(1:24) = x;   
    Xnew    = X;
end

%% 目标函数,修改obj1、obj2即可
% 最小配置200MW/400MWh
function Obj = fobj(x, Storage, P_load, Price, Renewable)
    %计算电网交互Grid值，Grid>0从电网买电，Grid<0
    buy_price   = Price(1, :);   % 电网买电价格，单位：元/kWh 
    sell_price  = Price(2, :);   % 储能售电价格，单位：元/kWh 
    lease_price = Price(3, 1);   % 容量租赁收益单价，108 元/kWh
    dr_price    = Price(4, 1);   % 需求响应价格，2 元/kWh
    model_price = Price(5, 1);   % 弃风弃光充电价格，单位：元/kWh
    
    Grid = P_load - (x(25:48) + x(1:24));
    if min(Grid) >= 0   % 从电网买电,不返送电上网
        %计算一天的净收益
        netgain = 0;
        for i = 1:24
            if x(i) < 0   %为负代表冲电
                if x(i+24) <= P_load(i)
                    netgain = netgain + x(i)*1*buy_price(i);
                else
                    delta(i) = Renewable(i) - x(i+24);
                    netgain = netgain - delta(i)*model_price - Grid(i)*buy_price(i);
                end
            else  %为正代表放电
                netgain = netgain + x(i)*1*sell_price(i) ;   % x(i)（+）为储能放出电量的功率，*1h*卖电价即为卖电收益
            end
        end
                          
        % 计算一年的总收益,一年运行330天，系统成本880元/kWh， 1%的年运维成本，-弃风弃光成本，+容量租赁收益
        % 宁德时代-传统储能-构网型：系统成本590元/kWh，运维成本12元/kWh，更换成本380元/kWh，12年更换，
        Netgain_total = 330*netgain - 590*Storage/25 - 12*Storage + Storage*lease_price - 0*Storage;   %26年
%         Netgain_total = 330*netgain - 590*Storage/25 - 12*Storage + Storage*lease_price - 380/12*Storage;

        Obj = Netgain_total/(590*Storage/25);     
    else
        Obj = -10000;   %该组粒子会返送电，将收益降到极大，保证该组粒子结果不会被选中
    end     
end






















