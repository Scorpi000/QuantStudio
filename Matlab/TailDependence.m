function [TD_low,TD_up] = TailDependence(x,y,copula_type,kenel)
% 根据核密度估计求x和y的边际分布
[U,~] = ksdensity(x,x,'function','cdf','kernel',kenel);
[V,~] = ksdensity(y,y,'function','cdf','kernel',kenel);
% 估计 Copula 函数的参数, 并计算尾部相关性
if isequal(copula_type,'t')
    [rho_t, nuhat] = copulafit('t',[U(:),V(:)]);
    k = round(nuhat);
    rho = rho_t(1,2);
    alpha = ((1+k) * (1-rho) / (1+rho))^0.5;
    TD_up = 2 - 2 * tcdf(alpha, k+1);
    TD_low = TD_up;
elseif isequal(copula_type,'Clayton')
    rho = copulafit('Clayton',[U(:),V(:)]);
    TD_up = 0;
    TD_low = 2^(-1/rho);
elseif isequal(copula_type,'Gumbel')
    rho = copulafit('Gumbel',[U(:),V(:)]);
    TD_up = 2-2^(1/rho);
    TD_low = 0;
else
    TD_up = 0;
    TD_low = 0;
end
end