%三维无人机航迹优化

%适应度函数
function [fitness,SuccessFlag] = fun(position,NodesNumber,startPoint,endPoint,ThreatAreaPostion,ThreatAreaRadius)

%判断路径是否OK
[SuccessFlag]=IsPathOk(position,NodesNumber,startPoint,endPoint,ThreatAreaPostion,ThreatAreaRadius);
if SuccessFlag<1 %如果路径不OK，采用惩罚值
    fitness = 10E32;
else
    %获取插值后的路径
    [X_seq,Y_seq,Z_seq,x_seq,y_seq,z_seq] = GetThePathLine(position,NodesNumber,startPoint,endPoint); 
    %% 计算三次样条得到的离散点的路径长度（适应度）
    dx = diff(X_seq);
    dy = diff(Y_seq);
    dz = diff(Z_seq);

    PathLength = sum(sqrt(dx.^2 + dy.^2 + dz.^2));%路径长度

    Height = sum(((Z_seq - mean(Z_seq)).^2).^0.5); %高度评价

    Dx = diff(x_seq');
    Dy = diff(y_seq');
    Dz = diff(z_seq');

    for i = 1:size(Dx,2)-1
       C(i) = (Dx(i)*Dx(i+1)+Dy(i)*Dy(i+1)+Dz(i)*Dz(i+1))/(sqrt(Dx(i)^2+Dy(i)^2+Dz(i)^2)*sqrt(Dx(i+1)^2+Dy(i+1)^2+Dz(i+1)^2)) ;
    end

    phi = pi/2;
    Curve = sum(cos(phi)-C);  %转弯角评价
    %设置权重
    w1 = 0.4;
    w2 = 0.4;
    w3 = 0.2;
    fitness = w1*PathLength+w2*Height + w3*Curve;
    
end

end