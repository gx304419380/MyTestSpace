function V_bus = Func_power_flow(x, Location_Storage, P_load, P_new, Bus, Branch)
% ����ʦ����
% ���룺x�����ܳ�ŵ繦�ʣ�P_load�����ɣ�P_new��������
% ���룺Bus����һ�У��ڵ�ţ��ڶ��У������й���kW���������У������޹�(kvar)
% ���룺Branch����1�У�֧·�ţ���2�У�֧·�׽ڵ㣻��3�У�֧·β�ڵ㣻
%               ��4�У�֧·���裻��5�У�֧·�翹 ��ŷķ��
% �����V_bus���ڵ��ѹ

% ����33�ڵ�����
SB = 1;     %��λMVA
UB = 230;   %ϵͳ���ѹ����λkV��IEEE33Ϊ12.66kV

%����
pload_flex = P_load / 3715;
Bus(:,2) = Bus(:,2)*pload_flex;   %���ɲ����仯
Bus(:,3) = Bus(:,3)*pload_flex;

WEI_new = [3, 6];  %��硢��������λ�ã��ȷ�磬����
for kk = 1:length(WEI_new)
    Bus(WEI_new(kk),2) = Bus(WEI_new(kk),2)-P_new(kk);
    Bus(WEI_new(kk),3) = Bus(WEI_new(kk),3)-P_new(kk)*0.484;
end

WEI = Location_Storage;   %���ܽ���λ��
for kk=1:length(WEI)
    Bus(WEI(kk),2)= Bus(WEI(kk),2)-x(kk);
    Bus(WEI(kk),3)= Bus(WEI(kk),3)-x(kk)*0.484;
end

Bus(:,2) = Bus(:,2)/1000/SB;
Bus(:,3) = Bus(:,3)/1000/SB;

Branch(:,4) = Branch(:,4)*SB/UB^2;
Branch(:,5) = Branch(:,5)*SB/UB^2;


%���õ�ѹ��ʼֵ
[busnum,dump1] = size(Bus);
Vbus = ones(busnum,1);     
Vbus(1,1) = 1;%ƽ��ڵ��ѹ
cita = zeros(busnum,1);
[branchnum,dump2] = size(Branch);

k = 0;  %��������
Ploss = zeros(branchnum,1);%��֧·���й����  
Qloss = zeros(branchnum,1);%֧·�޹����
P = zeros(branchnum,1);%��֧·���й�  
Q = zeros(branchnum,1);%֧·�޹�
I = zeros(branchnum,1);
F = 0;%����������־
%%���γ���֧·��������   s1Ϊ�ź����֧·����
TempBranch=Branch;
n=1;
s2=[];         %�����������֧·Ӧ��û������
while ~isempty(TempBranch)   %�ж��Ƿ�Ϊ��
    [s,dump3]=size(TempBranch);%sΪ֧·��  
    m=1;
    while s>0
       i=find(TempBranch(:,2)==TempBranch(s,3));%ĩ�˽ڵ��Ƿ�Ϊ����֧·�׶˽ڵ�
        if isempty(i)
            s1(n,:)=TempBranch(s,:);%���i�ǿռ���ýڵ�ΪҶ�ڵ�
            n=n+1;
        else s2(m,:)=TempBranch(s,:);%���i���ǿռ���ýڵ�Ϊ��Ҷ�ڵ�
            m=m+1;
        end
        s=s-1;
   end
    TempBranch=s2;             %��s2��ֵ��TempBranch.�ظ������жϣ�ֱ��TempBranchΪ�ռ�Ϊֹ
    s2=[];
end
    %ǰ�ƽ���֧·���ʼ���
    Bus1=Bus;
    while (k<100)&&(F==0)  
    Pij1=zeros(busnum,1); %��֧·�׶˽ڵ㼰������������������֧·�Ĺ������
    Qij1=zeros(busnum,1);
   
    
      
    for s=1:branchnum        
        ii=s1(s,2);      %ȡ�ź����֧·�׽ڵ�   
        jj=s1(s,3);      %ȡ�ź����֧·β�ڵ�  
        Pload=Bus(jj,2);
        Qload=Bus(jj,3); %�ڵ��й����޹�����
        R=s1(s,4);
        X=s1(s,5);
        VV=Vbus(jj,1);
        Pij0=Pij1(jj); %��֧·ĩ�˽ڵ�ĺ�������
        Qij0=Qij1(jj);
        II=((Pload+Pij0)^2+(Qload+Qij0)^2)/(VV^2);    %֧·������ƽ��
        Ploss(s1(s,1))=II*R;                          %֧·�й����޹����    �����д��Ľ�������Ploss
        Qloss(s1(s,1))=II*X;
        P(s1(s,1))=Pload+Ploss(s1(s,1))+Pij0;       %֧·���ʣ����������ڵ㹦�ʺ��������
        Q(s1(s,1))=Qload+Qloss(s1(s,1))+Qij0;
        Pij1(ii)=Pij1(ii)+P(s1(s,1));               %֧·�׶˹��ʵ�λMW
        Qij1(ii)=Qij1(ii)+Q(s1(s,1));
    end
    %%���Ƽ���ڵ��ѹ
for s=branchnum:-1:1
    ii=s1(s,3); 
    kk=s1(s,2);
    R=s1(s,4);
    X=s1(s,5);
    Vbus(ii,1)=(Vbus(kk,1)-(P(s1(s,1))*R+Q(s1(s,1))*X)/Vbus(kk,1))^2+((P(s1(s,1))*X-Q(s1(s,1))*R)/Vbus(kk,1))^2;
    cita(ii,1)=cita(kk,1)-atan(((P(s1(s,1))*X-Q(s1(s,1))*R)/Vbus(kk,1))/(Vbus(kk,1)-(P(s1(s,1))*R+Q(s1(s,1))*X)/Vbus(kk,1)));
    Vbus(ii,1)=sqrt(Vbus(ii,1));%������ڵ��ѹ��ֵ
end
 

%    %�����ϲ���ѹǰ���ٴμ���֧·����
%     Pij2=zeros(busnum,1); 
%     Qij2=zeros(busnum,1);
%     
%  for s=1:branchnum  
%      ii=s1(s,2);        
%      jj=s1(s,3);
%      Pload=Bus(jj,2);
%      Qload=Bus(jj,3);%�ڵ��й����޹�����
%      R=s1(s,4);
%      X=s1(s,5);
%      VV=Vbus(jj,1);
%      Pij0=Pij2(jj);%��֧·ĩ�˽ڵ�ĺ�������
%      Qij0=Qij2(jj);
%      II=((Pload+Pij0)^2+(Qload+Qij0)^2)/(VV^2);%֧·������ƽ��
%      I(s1(s,1))=sqrt(II)*1000;                         %��λA
%      Ploss(s1(s,1))=II*R;                          %֧·�й����޹����  
%      Qloss(s1(s,1))=II*X;
%      P(s1(s,1))=Pload+Ploss(s1(s,1))+Pij0;       %֧·���ʣ����������ڵ㹦�ʺ��������
%      Q(s1(s,1))=Qload+Qloss(s1(s,1))+Qij0;
%      Pij2(ii)=Pij2(ii)+P(s1(s,1));           %֧·�׶˹���
%      Qij2(ii)=Qij2(ii)+Q(s1(s,1));
%  end
%      ddp=max(abs(Pij1(:,1)-Pij2(:,1)));
%      ddq=max(abs(Qij1(:,1)-Qij2(:,1)));
%      pr=1e-3;%����
%      
%      L1=(ddp<pr)&&(ddq<pr); 
%      F=L1;


     k=k+1;
 end
if k==10
    disp('������������')
    Pij2(1)=3.9;
end
    
%     P1=0;Q1=0;
%  for s=1:branchnum    %�������й���� �� ���޹����
%      P1=P1+Ploss(s);
%      Q1=Q1+Qloss(s);
%  end

 % �������ֵ���ڵ��ѹ�����Ϊ������
 if size(Vbus, 2) == 1
     V_bus = Vbus;
 else
     V_bus = Vbus';
 end
 

 
%  disp('�ڵ��ѹ��ֵ')
%  Vbus
%  disp('�ڵ��ѹ��ǣ��ȣ�')
%  cita*180/pi
% disp('֧·��   �׽ڵ�   ĩ�ڵ�   ֧·����(kW)   ֧·���(kvar)')
% [Branch(:,1:3) (P+1i*Q)*1000*SB  (Ploss+1i*Qloss)*1000*SB]
%  disp('�����')
%  (P1+1i*Q1)*1000*SB
%  disp('����������')
%  k
 

