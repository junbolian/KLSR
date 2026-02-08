%三维无人机航迹优化

%% 根据节点，采用三次样条插值，平滑路径，并返回路径坐标
%[X_seq,Y_seq,Z_seq]为插值后的路径
%[x_seq,y_seq,z_seq]为插值前的路径
function [X_seq,Y_seq,Z_seq,x_seq,y_seq,z_seq] = GetThePathLine(postion,NodesNumber,startPoint,endPoint)
        
        for i = 1:NodesNumber
            postionP(i,1) = startPoint(1) + (endPoint(1)-startPoint(1))*i/(NodesNumber+1);
        end
        postionP(:,2) = postion(1:NodesNumber)';
        postionP(:,3) = postion(NodesNumber + 1:2*NodesNumber)';
        %根据X坐标排序
        [~,SortIndex]=sort(postionP(:,1));
        postionP(:,1) = postionP(SortIndex,1);
        postionP(:,2) = postionP(SortIndex,2);
        postionP(:,3) = postionP(SortIndex,3);
         % %对Z方向做处理，使得生成的z一定是在地面或者山峰以上。
        % for i = 1:size(postionP)
        for i = 1:size(postionP,1)
            x= postionP(i,2);
            y = postionP(i,1);
            postionP(i,3) = postionP(i,3) + MapValueFunction(x,y);
        end
        
        
        PALL = [startPoint;postionP;endPoint];%加入起始点
        x_seq=PALL(:,1);%所有节点的横坐标
        y_seq=PALL(:,2);
        z_seq=PALL(:,3);


        k = size(PALL,1);%起点+终点+中间节点
        i_seq = linspace(0,1,k);%用于产生x1,x2之间的N点行矢量。其中x1、x2、k分别为起始值、中止值、元素个数。若缺省N，默认点数为100。在matlab的命令窗口下输入help linspace或者doc linspace可以获得该函数的帮助信息。
        I_seq = linspace(0,1,200);
        %三次样条插值
        X_seq = spline(i_seq,x_seq,I_seq);
        Y_seq = spline(i_seq,y_seq,I_seq);
        Z_seq = spline(i_seq,z_seq,I_seq);
%         X_seq = interp1(i_seq,x_seq,I_seq);
%         Y_seq = interp1(i_seq,y_seq,I_seq);
%         Z_seq = interp1(i_seq,z_seq,I_seq);

end