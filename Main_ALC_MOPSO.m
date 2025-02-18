% 大唐项目储能项目-清洁能源基地新型储能优化选型与动态配置
% 从25年底光伏并网，26年底风电并网，全生命周期分析
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
n_sence     = 1;      % 仿真场景循环次数，1为单次分析，= N_sence=100为多场景分析
season      = '夏';   % 夏或冬季典型日
year        = 27;     % 26年只有475MW光伏，27年风光共925MW，从27年开始计算全生命周期最优储能配置
Alc_year    = 25;     % 储能使用年限

% 以下为默认值，可修改
Rate_Load   = 1.5;    % 负荷运行与风光出力比值
Rate_Power  = 1.0;    % 风光出力倍率
Rate_limt   = [0.85; 1]; %新能源消纳率约束，95%-100%
Storage_max = 300;  %最大储能额定容量，MW
Storage_min = 100;  %最小储能额定容量，MW
Max_Iter    = 50;    %最大迭代次数
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

N_obj      = 1;      %优化目标数
dim        = 24*Alc_year;
% 粒子维度，前24维日时序功率，25-48风光实际出力，49维储能容量
if N_obj == 2
    x3_limt   = [10, 18];          %储能接入的节点限制
    % 导入网架结构，电力潮流
    % 第一列：节点号，第二列：负荷有功（kW），第三列：负荷无功(kvar)
    Bus    = struct2array(load('Bus_IEEE33.mat'));
    % 第1列存支路号，第2列存支路首节点，第3列存支路尾节点，第4存支路电阻，第5列存支路电抗 （欧姆）
    Branch = struct2array(load('Branch_IEEE33.mat'));
end

%% 输入价格
sell_price  = ones(1, 24) * 0.538;   %单位：0.488 元/kWh
lease_price = ones(1, 24) * 108;     %容量租赁收益单价，108 元/kWh
data_price  = xlsread('Data_Input.xlsx', 'Price');    %价格
buy_price   = data_price(:,1)';   %单位：元/kWh
dr_price    = data_price(:,4)';   %需求响应价格，2 元/kWh
model_price = 0;                  %弃风弃光充电价格，元/kWh
Price       = [buy_price; sell_price; lease_price; dr_price; model_price*ones(1,24)];

%% 输入新能源规划
% 风光装机容量数据来源：《储能配置方案报告_2024.10.21》
Source_Plan = xlsread('Data_Input.xlsx', '风光规划', 'B3:C27')';   % B2:C26

% *********************风光出力输入及其不确定性输入*********************** %
Capacity_WT  = Source_Plan(1, :);   %风电装机容量，45万千瓦，26年底并网
Capacity_PV  = Source_Plan(2, :);   %光伏装机容量，47.5万千瓦，25年底并网
if strcmpi(season, '夏') == 1
    Source_coef = xlsread('Data_Input.xlsx', '风光系数', 'A2:B25')';   %夏季出力系数，行向量
else
    Source_coef = xlsread('Data_Input.xlsx', '风光系数', 'C2:D25')';   %冬季出力系数，行向量
end
% 出力系数根据《河池地区光伏项目15分钟钟出力(常吉、盛景、拉才)》计算
for i = 1:Alc_year
    WT(i,:) = 10000*Capacity_WT(i).*Source_coef(1,:);   %风电出力，单位从万千瓦转成kW
    PV(i,:) = 10000*Capacity_PV(i).*Source_coef(2,:);   %光伏出力，单位从万千瓦转成kW
    Renewable(i,:) = PV(i,:) + WT(i,:);                 %新能源=光伏+风电，行向量
end
WT        = WT .* Rate_Power;
PV        = PV .* Rate_Power;
Renewable = Renewable .* Rate_Power;                    %风光出力倍率
% ---------------------风光出力输入及其不确定性结束----------------------- %

% ***********************负荷输入及其不确定性输入************************* %
% 数据来源：《河池南丹（天峨）工业园源网荷储一体化项目前期负荷调查报告（10月份） (11.11-y改v3)》
% 根据风光出力计算负荷，按照出力电量：负荷电量=1：1.5为基准计算
Load_access = ones(1, 1);   %负荷接入,数量8，1表示接入，0表示不接入，
Uncertainty = '正态';  %不确定性分布，正态分布或均匀分布

if length(Load_access) == 1
    %根据出力和负荷配比计算负荷
    Load_Coef = xlsread('Data_Input.xlsx', '所有负荷', 'B7:C30');   %负荷系数
    if strcmpi(season, '夏') == 1
        Load_coef = Load_Coef(:, 2:2:end);     %夏季负荷系数
        Load_max  = ones(Alc_year, 1) .* sum(Renewable,2) / sum(Load_coef) /10000 /0.85;
    else
        Load_coef = Load_Coef(:, 1:2:end-1);   %冬季负荷系数
        Load_max  = ones(Alc_year, 1) .* sum(Renewable,2) / sum(Load_coef) /10000 /0.85;
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
for i = 1:Alc_year
    P_load(i,:) = 10000*sum(Load_access.* Load_max(i, :).*Load_coef, 2)'.*0.85;
    P_load(i,:) = P_load(i,:) .* Rate_Load;   %
    Rate_Load_cal(i) = sum(P_load(i,:)) / sum(Renewable(i,:));
end
% -----------------------负荷输入及其不确定性结束------------------------- %

SOC_0       = 0.3;   %储能系统初始SOC
SOC_limt    = [0.1; 1];     %SOC约束，10%-100%
%% 考虑概率因素的储能配置优化,胜场源荷不确定仿真场景
% 根据3西格玛原则，数据分布在-+3sigma范围内的概率为99.73%
% 因此根据不确定范围计算后续数据统计分布的标准差

N_s    = 1000*ceil(n_sence/100);   %生成场景数
N_sence = 100*ceil(n_sence/100);   %缩减场景数，缩减至100个
% LHS生成源荷不确定性的仿真场景
Load_all = [];   %1000*(24*Alc_year)，日负荷仿真场景
EnWT_all = [];   %1000*(24*Alc_year)，日风电出力仿真场景
EnPV_all = [];   %1000*(24*Alc_year)，日光伏出力仿真场景
Ener_all = [];   %1000*(24*Alc_year)，日风光出力仿真场景
if strcmpi(Uncertainty, '正态') == 1
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
% 采用下采样减少场景数，从1000缩减到100
Load_All = [];   %100*(24*Alc_year)，每行为日负荷时间序列
EnWT_All = [];   %100*(24*Alc_year)，每行为日风电出力时间序列
EnPV_All = [];   %100*(24*Alc_year)，每行为日光伏出力时间序列
Ener_All = [];   %100*(24*Alc_year)，每行为日出力时间序列
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

%% PSO算法参数
% 粒子维度，前24维日时序功率，25-48风光实际出力，49维储能容量
% 此处功率为储能放出的功率（+），和给储能充电的功率（-）
% 储能电池内的电量变化，放电为功率*1/0.93（+），充电为功率*1*0.93（-）

w_max      = 0.9;   %最大惯性系数
w_min      = 0.4;   %最小惯性系数
v1_max(1, 1:24) = Storage_min*0.1*1000*ones(1,24);     %速度上界
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

%% 考虑概率因素的储能配置优化，计算每个场景下的最优储能；
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
    Price_ns     = repmat(Price, 1, Alc_year);   %根据价格是否波动设定，生成当前场景价格
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

    %% PSO循环
    for i = 1:24
        if sell_price(i) > buy_price(i)
            X1_upper(1,i) = 0 ;           %千瓦，充放电参数0.5C，放电为(+)
            X1_lower(1,i) = -Storage_min/2;   %千瓦，充放电参数0.5C，充电为(-)
        else
            X1_upper(1,i) = Storage_min/2 ;   %千瓦，充放电参数0.5C，放电为(+)
            X1_lower(1,i) = -Storage_min/2;   %千瓦，充放电参数0.5C，充电为(-)      
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
    
    X1 = Func_initialization(N_pop, X1_upper, X1_lower, dim);   %初始化种群，功率
    X2 = Func_initialization(N_pop, X2_upper, X2_lower, dim);   %初始化种群，风光实际出力
    X3 = Func_initialization(N_pop, X3_upper, X3_lower, N_obj); %初始化种群，容量、位置
    X3(:, 1) = round(X3(:, 1)/10000)*10000;   %容量，最小尺度10MW
    if N_obj == 2
        X3(:, 2) = round(X3(:, 2));           %位置
    end
    
    V1 = Func_initialization(N_pop, v1_max, v1_min, dim);   %初始化种群速度
    V2 = Func_initialization(N_pop, v2_max, v2_min, dim);   %初始化种群速度
    V3 = Func_initialization(N_pop, v3_max, v3_min, N_obj);   %初始化种群速度
    
    for i = 1:N_pop
        Plim = pload_ns - X2(i,:);   %储能放电限制，避免向电网返送电
        [X1(i,:), X3(i,1)] = Func_ConstraintCheck(X1(i,:), X3(i, 1), SOC_limt, SOC_0, Plim);
    end

    fx = zeros(N_obj, N_pop);
    % 计算粒子目标函数：1、储能经济性，取最大；2、节点电压波动倒数，取最大
    % 计算粒子目标函数1，储能经济性，取最大
    for i = 1:N_pop
        Storage = X3(i, 1);
        fx(1, i) = Func_obj(X1(i,:), X2(i,:), X3(i, 1), pload_ns, Price_ns, renewable_ns);   %计算适应度值
    end     
    % 计算粒子目标函数2，节点电压波动倒数，取最大
    if N_obj == 2
        V_bus = [];   % 24列
        for k = 1:24
            P_new = [EnWT_All(ns, k), EnPV_All(ns, k)];
            Location_Storage = X(i, 50);   %储能位置
            v_bus = Func_power_flow(X(i, k), Location_Storage, pload_ns(k), P_new, Bus, Branch);
            V_bus = [V_bus, v_bus];
        end
        for k = 1:size(V_bus, 1)
            vbus = V_bus(k, :);
            temp_fx2(k) = sum(abs(vbus-mean(vbus)));
        end
        fx(2, i)  = 1/sum(temp_fx2);   % 1/平均节点电压波动率
    end
    
    pBest1   = X1;   % 将初始种群作为历史最优
    pBest2   = X2;
    pBest3   = X3;
    fx_pBest = fx;   % 记录初始全局最优解,默认优化最大值

    % 记录初始全局最优解
    [~, index]  = max(sum(fx, 1));   %寻找适应度最大的位置    
    fx_gBest    = -inf;   %记录适应度值和位置
    gBest1      = X1(index(1),:);
    gBest2      = X2(index(1),:);
    gBest3      = X3(index(1),:);
    gBest_index = index(1);

    X1new     = X1;             % 新位置
    X2new     = X2;             % 新位置
    X3new     = X3;             % 新位置
    fx_New    = fx;              % 新位置适应度值
    
    %% ***********************PSO循环开始************************* %
    for t = 1:Max_Iter
        wait = ceil(100*ns/n_sence)-1 + ceil(100*t/Max_Iter)/100;
        wait_str = ['仿真场景循环完成 ', num2str(wait), '%'];
        waitbar(wait/100, hwait, wait_str);
        % 计算种群适应度
        for i = 1:N_pop
            w  = w_max-(w_max-w_min)*(t)^2/(Max_Iter)^2;   %惯性权重更新
            c1 = (0.5-2.5)*t/Max_Iter+2.5;                 %加速因子c1更新
            c2 = (2.5-0.5)*t/Max_Iter+0.5;                 %加速因子c2更新
            r1 = rand(1,dim);
            r2 = rand(1,dim);
            r3 = rand(1,N_obj);
            r4 = rand(1,N_obj);           
            V1(i,:) = w*V1(i,:) + c1.*r1.*(pBest1(i,:) - X1(i,:)) + c2.*r2.*(gBest1 - X1(i,:));
            V2(i,:) = w*V2(i,:) + c1.*r1.*(pBest2(i,:) - X2(i,:)) + c2.*r2.*(gBest2 - X2(i,:));
            V3(i,:) = w*V3(i,:) + c1.*r3.*(pBest3(i,:) - X3(i,:)) + c2.*r4.*(gBest3 - X3(i,:));
            V1(i,:) = Func_BoundaryCheck(V1(i,:), v1_max, v1_min, dim);   %速度边界检查及约束
            V2(i,:) = Func_BoundaryCheck(V2(i,:), v2_max, v2_min, dim);   %速度边界检查及约束
            V3(i,:) = Func_BoundaryCheck(V3(i,:), v3_max, v3_min, N_obj);   %速度边界检查及约束
            X1new(i,:) = X1(i,:) + V1(i,:);                           %位置更新
            X2new(i,:) = X2(i,:) + V2(i,:);
            X3new(i,:) = X3(i,:) + V3(i,:);
            
            % ----------------粒子边界检测及约束------------------ %
            X2new(i,:) = Func_BoundaryCheck(X2new(i,:), X2_upper, X2_lower, dim);
            X3new(i,:) = Func_BoundaryCheck(X3new(i,:), X3_upper, X3_lower, N_obj);
            if N_obj == 2
                X3new(i,2) = round(X3new(i,2));   %储能位置
            end
            X3new(i, 1)  = round(X3new(i, 1)/10000)*10000;   %容量，最小尺度10MW
            
            Storage = X3new(i, 1);
            Plim    = pload_ns - X2new(i,:);   %储能放电限制，避免向电网返送电
            [X1new(i,:), X3new(i,1)] = Func_ConstraintCheck(X1new(i,:), Storage, SOC_limt, SOC_0, Plim);

            Grid(:, i) = (pload_ns - (X1new(i,:) + X2new(i,:)))'./1000;
            if min(Grid(:, i)) < 0
                Grid_index(1, i) = -1;
            else
                Grid_index(1, i) = 1;
            end
            
            % 计算粒子目标函数：1，储能经济性，取最大；2，节点电压波动倒数，取最大
            % 计算粒子目标函数1，储能经济性，取最大
            for i = 1:N_pop
                Storage = X3(i, 1);
                fx(1, i) = Func_obj(X1(i,:), X2(i,:), X3(i, 1), pload_ns, Price_ns, renewable_ns);   %计算适应度值
            end     
            % 计算粒子目标函数2，节点电压波动倒数，取最大
            if N_obj == 2
                V_bus = [];   % 24列
                for k = 1:24
                    P_new = [EnWT_All(ns, k), EnPV_All(ns, k)];
                    Location_Storage = X(i, 50);   %储能位置
                    v_bus = Func_power_flow(X(i, k), Location_Storage, pload_ns(k), P_new, Bus, Branch);
                    V_bus = [V_bus, v_bus];
                end
                for k = 1:size(V_bus, 1)
                    vbus = V_bus(k, :);
                    temp_fx2(k) = sum(abs(vbus-mean(vbus)));
                end
                fx(2, i)  = 1/sum(temp_fx2);   % 1/平均节点电压波动率
            end
            
             % 更新历史最优值，目标函数等权重
            if sum(fx_New(:, i)) >= sum(fx_pBest(:, i))
                pBest1(i,:)    = X1new(i,:);
                pBest2(i,:)    = X2new(i,:);
                pBest3(i,:)    = X3new(i,:);
                fx_pBest(:, i) = fx_New(:, i);    
            end
            % 更新全局最优值，两个目标函数等权重
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
        % 记录当前迭代最优值和最优适应度值
        IterCurve(t, :) = fx_gBest;    %记录当前迭代的最优解适应度值
    end
    %% -----------------------PSO循环结束------------------------- %        
    Result_All{1, ns} = X1./1000;
    Result_All{2, ns} = X2./1000;
    Result_All{3, ns} = X3./1000;
    
    temp_gBest1 = gBest1'./1000;
    temp_gBest2 = gBest2'./1000;
    temp_gBest3 = gBest3'./1000;
    if N_obj == 2
        temp_gBest3(:,2).*1000;
    end
    
    Max_Pdelta = max(renewable_ns - pload_ns)/1000;  % 风光出力-负荷最大差值，MW
    Max_Pstor = max(abs(temp_gBest1));               % 最大充放电功率，MW
    Rate = sum(temp_gBest2)/sum(renewable_ns./1000);  %消纳率，%
    Elec_rate = sum(temp_gBest2)/sum(pload_ns/1000);  %新能源消纳占比，%
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

%% 输出结果
P_stor_opt  = Result_Opt1(1:end, Opt_ns)';     %储能24小时功率，kW
disp('储能容量优化结果：功率MW/容量MWh')
max_Pstor = max(abs(P_stor_opt));
Storage_opt = Result_Opt(1, Opt_ns);
[max_Pstor, Storage_opt]
disp('出力-负荷最大差，风光出力需存容量（MWh）')
max_Pdelta = max(Ener_All(Opt_ns, :)/1000 - Load_All(Opt_ns, :)/1000);
plim = Load_All(Opt_ns, :) - Ener_All(Opt_ns, :);
Sum_Pchr = -sum(plim(find(plim<0)))/1000/Alc_year;
[max_Pdelta, Sum_Pchr]
disp('消纳率，新能源消纳占比（%）；用电量，发电量，购电量（亿千瓦时）')
rate      = sum(Result_Opt2(:, Opt_ns))/sum(Ener_All(Opt_ns, :)/1000);
elec_rate = sum(Result_Opt2(:, Opt_ns))/sum(Load_All(Opt_ns, :)/1000);
Elec_generation  = sum(Result_Opt2(:, Opt_ns)*1000)*330/100000000/Alc_year;
Elec_consumption = sum(sum(P_load))*330/100000000/Alc_year;
Elec_buy = Elec_consumption - Elec_generation;
[rate, elec_rate, Elec_consumption, Elec_generation, Elec_buy]
Result = Result_All{1, Opt_ns};


%% 优化结果画图
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
legend('负荷', '风光出力')
xlim([be-1, en])
xlabel('t/h')
ylabel('MW')


figure(2)
% 最优容量下对应的日功率曲线
stairs(P_stor_opt(be:en), '-s','Color',[1 0.64706 0], ...
     'MarkerEdgeColor', 'k','MarkerFaceColor','r','LineWidth', 2)
hold on
line([0,24*Alc_year], [0,0],'linestyle','--','color','r')
xlim([be-1, en])
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
x    = P_stor_opt(1:24);
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
    ylabel('频数')
    title('频数分布直方图')
end

figure(6)
stairs(Grid_opt, '-s','Color',[1 0.64706 0], ...
     'MarkerEdgeColor', 'k','MarkerFaceColor','r','LineWidth', 2)
hold on
line([0,24*Alc_year], [0,0],'linestyle','--','color','r')
% xlim([0, 24])
xlabel('t/h')
ylabel('功率/MW')
title('电网购电')


%% 后续为函数

%% 粒子群初始化函数,初始化位置或速度
function X = Func_initialization(pop, ub, lb, dim)
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
function X = Func_BoundaryCheck(x, ub, lb, dim)
    % dim为数据的维度大小
    % x为输入数据，维度为[1,dim];
    % ub为数据上边界，维度为[1,dim]
    % lb为数据下边界，维度为[1,dim]
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


%% 储能充放电功率约束条件
function [X1new, X3new] = Func_ConstraintCheck(X, Storage, SOC_limt, SOC_0, Plim)
    % dim为数据的维度大小
    % x为输入数据，维度为[1,dim];
    % SOC_limt：SOC上下界约束，[0.1, 1]
    % Rate_limt：消纳率上下界约束，[0.95, 1]
    SOC     = [];
    SOC_re  = [];
    SOC_min = Storage*SOC_limt(1);
    SOC_max = Storage*SOC_limt(2);
    X_3 = Storage;
    eta_chg = 0.927;   % 充电效率
    eta_dc  = 0.928;   % 放电效率
    % 充放电约束思路：将充放电功率转换到电池的电量变化w（增加或减少）；日增减和为0；
    % w的累加计算SOC，w的正负约束充放电时刻。
    
    for k = 1:length(X)/24
        % 约束每日充放电量相等
        ii1 = 24 *(k-1) + 1;
        ii2 = 24 * k;
        x = X(ii1:ii2);
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
                x3(k) = ceil(x49/10000)*10000;   %容量，最小尺度10MW
                if x3(k) > X_3
                    X_3 = x3(k);
                end
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
        xnew(ii1:ii2) = x;
    end
    
    X1new = xnew;
    X3new = X_3;
end


% 目标函数
function Obj = Func_obj(x1, x2, Storage, P_load, Price, Renewable)
    
    % 计算电网交互Grid值，Grid>0从电网买电，Grid<0
    buy_price   = Price(1, :);   % 电网买电价格，单位：元/kWh 
    sell_price  = Price(2, :);   % 储能售电价格，单位：元/kWh 
    lease_price = Price(3, 1);   % 容量租赁收益单价，108 元/kWh
    dr_price    = Price(4, 1);   % 需求响应价格，2 元/kWh
    model_price = Price(5, 1);   % 弃风弃光充电价格，单位：元/kWh
    
    Grid = P_load - (x1 + x2);
    if min(Grid) >= 0   % 从电网买电,不返送电上网
        %计算一天的净收益
        netgain = 0;
        for i = 1:length(Grid)
            if x1(i) < 0   %为负代表冲电
                if x2(i) <= P_load(i)
                    netgain = netgain + x1(i)*1*buy_price(i);
                else
                    delta(i) = Renewable(i) - x2(i);
                    netgain = netgain - delta(i)*model_price - Grid(i)*buy_price(i);
                end
            else   %为正代表放电
                netgain = netgain + x1(i)*1*sell_price(i) ;   % x(i)（+）为储能放出电量的功率，*1h*卖电价即为卖电收益
            end
        end
                          
        %计算一年的总收益,一年运行330天，系统成本590元/kWh， 1%的年运维成本,-弃风弃光成本，+容量租赁收益-更换成本
        % 宁德时代-传统储能-构网型：系统成本590元/kWh，运维成本12元/kWh，更换成本380元/kWh，12年更换，
        Netgain_total = 330*netgain - 590*Storage/25 - 12*Storage + Storage*lease_price - 380/12*Storage;

        Obj = 25*Netgain_total/(590*Storage/25);     
    else
        Obj = -1000000;   %该组粒子会返送电，将收益降到极大，保证该组粒子结果不会被选中
    end  
end






















