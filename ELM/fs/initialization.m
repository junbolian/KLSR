%%  闲鱼：深度学习与智能算法
%%  唯一官方店铺：https://mbd.pub/o/author-aWWbm3BtZw==
%%  微信公众号：强盛机器学习，关注公众号获得更多免费代码！
function X=initialization(N,Dim,UB,LB)

B_no= size(UB,2); % numnber of boundaries

if B_no==1
    X=rand(N,Dim).*(UB-LB)+LB;
end

% If each variable has a different lb and ub
if B_no>1
    for i=1:Dim
        Ub_i=UB(i);
        Lb_i=LB(i);
        X(:,i)=rand(N,1).*(Ub_i-Lb_i)+Lb_i;
    end
    end
%%  闲鱼：深度学习与智能算法
%%  唯一官方店铺：https://mbd.pub/o/author-aWWbm3BtZw==
%%  微信公众号：强盛机器学习，关注公众号获得更多免费代码！
