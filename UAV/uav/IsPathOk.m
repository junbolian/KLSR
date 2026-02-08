%三维无人机航迹优化
%% 判断地图是否可行，即有没有穿过山峰或者威胁区域
function [SuccessFlag]=IsPathOk(postion,NodesNumber,startPoint,endPoint,ThreatAreaPostion,ThreatAreaRadius)
        
        SuccessFlag = 1;
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
        x_seq=PALL(:,1);
        y_seq=PALL(:,2);
        z_seq=PALL(:,3);


        k = size(PALL,1);
        i_seq = linspace(0,1,k);
        I_seq = linspace(0,1,200);
         %三次样条插值
        X_seq = spline(i_seq,x_seq,I_seq);
        Y_seq = spline(i_seq,y_seq,I_seq);
        Z_seq = spline(i_seq,z_seq,I_seq);
%         X_seq = interp1(i_seq,x_seq,I_seq);
%         Y_seq = interp1(i_seq,y_seq,I_seq);
%         Z_seq = interp1(i_seq,z_seq,I_seq);
        path = [X_seq', Y_seq', Z_seq'];
        % 判断生成的曲线是否与与障碍物相交
        for i = 1:size(path,1)
            x = path(i,1);
            y = path(i,2);
            z_interp = MapValueFunction(y,x);
            for k = 1:size(ThreatAreaPostion,1)
                if ((x-ThreatAreaPostion(k,1))^2 + (y-ThreatAreaPostion(k,2))^2)^0.5<ThreatAreaRadius(k)
                    SuccessFlag = 0;
                    break;
                end
            end
            if path(i,3) < z_interp
                SuccessFlag = 0;
                break;
            end
        end
end