function V_bus = Func_power_flow(x, Location_Storage, P_load, P_new, Bus, Branch)
% 邓老师程序
% 输入：x，储能充放电功率；P_load，负荷；P_new，风光出力
% 输入：Bus，第一列，节点号；第二列，负荷有功（kW）；第三列，负荷无功(kvar)
% 输入：Branch，第1列，支路号；第2列，支路首节点；第3列，支路尾节点；
%               第4列，支路电阻；第5列，支路电抗 （欧姆）
% 输出：V_bus，节点电压

% 采用33节点算例
SB = 1;     %单位MVA
UB = 230;   %系统额定电压，单位kV，IEEE33为12.66kV

%归算
pload_flex = P_load / 3715;
Bus(:,2) = Bus(:,2)*pload_flex;   %负荷波动变化
Bus(:,3) = Bus(:,3)*pload_flex;

WEI_new = [3, 6];  %风电、光伏接入的位置，先风电，后光伏
for kk = 1:length(WEI_new)
    Bus(WEI_new(kk),2) = Bus(WEI_new(kk),2)-P_new(kk);
    Bus(WEI_new(kk),3) = Bus(WEI_new(kk),3)-P_new(kk)*0.484;
end

WEI = Location_Storage;   %储能接入位置
for kk=1:length(WEI)
    Bus(WEI(kk),2)= Bus(WEI(kk),2)-x(kk);
    Bus(WEI(kk),3)= Bus(WEI(kk),3)-x(kk)*0.484;
end

Bus(:,2) = Bus(:,2)/1000/SB;
Bus(:,3) = Bus(:,3)/1000/SB;

Branch(:,4) = Branch(:,4)*SB/UB^2;
Branch(:,5) = Branch(:,5)*SB/UB^2;


%设置电压初始值
[busnum,dump1] = size(Bus);
Vbus = ones(busnum,1);     
Vbus(1,1) = 1;%平衡节点电压
cita = zeros(busnum,1);
[branchnum,dump2] = size(Branch);

k = 0;  %迭代次数
Ploss = zeros(branchnum,1);%存支路的有功损耗  
Qloss = zeros(branchnum,1);%支路无功损耗
P = zeros(branchnum,1);%存支路的有功  
Q = zeros(branchnum,1);%支路无功
I = zeros(branchnum,1);
F = 0;%迭代收敛标志
%%本段程序将支路重新排序   s1为排好序的支路矩阵
TempBranch=Branch;
n=1;
s2=[];         %这段重新排列支路应该没有问题
while ~isempty(TempBranch)   %判断是否为空
    [s,dump3]=size(TempBranch);%s为支路数  
    m=1;
    while s>0
       i=find(TempBranch(:,2)==TempBranch(s,3));%末端节点是否为其他支路首端节点
        if isempty(i)
            s1(n,:)=TempBranch(s,:);%如果i是空集则该节点为叶节点
            n=n+1;
        else s2(m,:)=TempBranch(s,:);%如果i不是空集则该节点为非叶节点
            m=m+1;
        end
        s=s-1;
   end
    TempBranch=s2;             %将s2赋值给TempBranch.重复上述判断，直到TempBranch为空集为止
    s2=[];
end
    %前推进行支路功率计算
    Bus1=Bus;
    while (k<100)&&(F==0)  
    Pij1=zeros(busnum,1); %该支路首端节点及与其相连的其他后续支路的功率情况
    Qij1=zeros(busnum,1);
   
    
      
    for s=1:branchnum        
        ii=s1(s,2);      %取排好序的支路首节点   
        jj=s1(s,3);      %取排好序的支路尾节点  
        Pload=Bus(jj,2);
        Qload=Bus(jj,3); %节点有功和无功负荷
        R=s1(s,4);
        X=s1(s,5);
        VV=Vbus(jj,1);
        Pij0=Pij1(jj); %该支路末端节点的后续功率
        Qij0=Qij1(jj);
        II=((Pload+Pij0)^2+(Qload+Qij0)^2)/(VV^2);    %支路电流的平方
        Ploss(s1(s,1))=II*R;                          %支路有功和无功损耗    这里有待改进。显现Ploss
        Qloss(s1(s,1))=II*X;
        P(s1(s,1))=Pload+Ploss(s1(s,1))+Pij0;       %支路功率，包括后续节点功率和网络损耗
        Q(s1(s,1))=Qload+Qloss(s1(s,1))+Qij0;
        Pij1(ii)=Pij1(ii)+P(s1(s,1));               %支路首端功率单位MW
        Qij1(ii)=Qij1(ii)+Q(s1(s,1));
    end
    %%回推计算节点电压
for s=branchnum:-1:1
    ii=s1(s,3); 
    kk=s1(s,2);
    R=s1(s,4);
    X=s1(s,5);
    Vbus(ii,1)=(Vbus(kk,1)-(P(s1(s,1))*R+Q(s1(s,1))*X)/Vbus(kk,1))^2+((P(s1(s,1))*X-Q(s1(s,1))*R)/Vbus(kk,1))^2;
    cita(ii,1)=cita(kk,1)-atan(((P(s1(s,1))*X-Q(s1(s,1))*R)/Vbus(kk,1))/(Vbus(kk,1)-(P(s1(s,1))*R+Q(s1(s,1))*X)/Vbus(kk,1)));
    Vbus(ii,1)=sqrt(Vbus(ii,1));%计算出节点电压幅值
end
 

%    %利用上步电压前推再次计算支路功率
%     Pij2=zeros(busnum,1); 
%     Qij2=zeros(busnum,1);
%     
%  for s=1:branchnum  
%      ii=s1(s,2);        
%      jj=s1(s,3);
%      Pload=Bus(jj,2);
%      Qload=Bus(jj,3);%节点有功和无功负荷
%      R=s1(s,4);
%      X=s1(s,5);
%      VV=Vbus(jj,1);
%      Pij0=Pij2(jj);%该支路末端节点的后续功率
%      Qij0=Qij2(jj);
%      II=((Pload+Pij0)^2+(Qload+Qij0)^2)/(VV^2);%支路电流的平方
%      I(s1(s,1))=sqrt(II)*1000;                         %单位A
%      Ploss(s1(s,1))=II*R;                          %支路有功和无功损耗  
%      Qloss(s1(s,1))=II*X;
%      P(s1(s,1))=Pload+Ploss(s1(s,1))+Pij0;       %支路功率，包括后续节点功率和网络损耗
%      Q(s1(s,1))=Qload+Qloss(s1(s,1))+Qij0;
%      Pij2(ii)=Pij2(ii)+P(s1(s,1));           %支路首端功率
%      Qij2(ii)=Qij2(ii)+Q(s1(s,1));
%  end
%      ddp=max(abs(Pij1(:,1)-Pij2(:,1)));
%      ddq=max(abs(Qij1(:,1)-Qij2(:,1)));
%      pr=1e-3;%精度
%      
%      L1=(ddp<pr)&&(ddq<pr); 
%      F=L1;


     k=k+1;
 end
if k==10
    disp('潮流不收敛！')
    Pij2(1)=3.9;
end
    
%     P1=0;Q1=0;
%  for s=1:branchnum    %计算总有功损耗 和 总无功损耗
%      P1=P1+Ploss(s);
%      Q1=Q1+Qloss(s);
%  end

 % 函数输出值，节点电压，输出为列向量
 if size(Vbus, 2) == 1
     V_bus = Vbus;
 else
     V_bus = Vbus';
 end
 

 
%  disp('节点电压幅值')
%  Vbus
%  disp('节点电压相角（度）')
%  cita*180/pi
% disp('支路号   首节点   末节点   支路功率(kW)   支路损耗(kvar)')
% [Branch(:,1:3) (P+1i*Q)*1000*SB  (Ploss+1i*Qloss)*1000*SB]
%  disp('总损耗')
%  (P1+1i*Q1)*1000*SB
%  disp('迭代次数：')
%  k
 

