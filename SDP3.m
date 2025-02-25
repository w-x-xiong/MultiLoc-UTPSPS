function [y_tilde] = SDP3(Rx, Ts, dTs, Qtau, c0)

%This MATLAB implementation reproduces the method proposed in 
%"Multistatic Localization With Unknown Transmitter Position and Signal Propagation Speed"

alpha = 500;

[~, N] = size(Ts);
[k, ~] = size(Rx);

Rg = c0*Ts;
dRg = c0*dTs;

b_vec = zeros(2*N, 1);

for j = 1:N
    b_vec(j) = (1/2)*(norm(Rx(:,j))^2 - Rg(1,j)^2);
    b_vec(N+j) = (1/2)*(norm(Rx(:,j))^2 - dRg(1,j)^2);
end

Ar_mtx = [];
Ad_mtx = [];

for j = 1:N
    Ar_mtx = [Ar_mtx;[Rx(:,j)',zeros(k,1)',-1,1/2,Rg(1,j)*Ts(1,j)*alpha,(Ts(1,j)*alpha)^2/2,-Rg(1,j),-Ts(1,j)*alpha]];
    Ad_mtx = [Ad_mtx;[zeros(k,1)',Rx(:,j)',0,-1/2,dRg(1,j)*dTs(1,j)*alpha,(dTs(1,j)*alpha)^2/2,0,0]];
end

A_mtx = [Ar_mtx',Ad_mtx']';

W_mtx = pinv(c0^2*Qtau);

Psi_mtx = [A_mtx'*W_mtx*A_mtx, -A_mtx'*W_mtx*b_vec; -b_vec'*W_mtx*A_mtx, b_vec'*W_mtx*b_vec];

cvx_begin

variables y(2*k+6) Y(2*k+6, 2*k+6)

expression obj

obj = trace([Y,y;y',1]*Psi_mtx);

minimize obj

subject to

y(2*k+1) == trace(Y(1:k,(k+1):2*k));

y(2*k+2) == trace(Y((k+1):2*k,(k+1):2*k));

y(2*k+4) == Y(2*k+3,2*k+3);

Y(2*k+5,2*k+5) == trace(Y(1:k,1:k)) - 2*y(2*k+1) + y(2*k+2);

y(2*k+6) == Y(2*k+3,2*k+5);

Y(2*k+3,2*k+6) == Y(2*k+4,2*k+5);

norm(y(1:k) - y((k+1):2*k)) <= y(2*k+5);

[Y,y;y',1] == semidefinite(2*k+7);

cvx_end

Br_mtx_diag_elem = [];
Bd_mtx_diag_elem = [];

for j = 1:N
    Br_mtx_diag_elem = [Br_mtx_diag_elem;norm(y(1:k) - Rx(:,j))];
    Bd_mtx_diag_elem = [Bd_mtx_diag_elem;norm(y(k+1:2*k) - Rx(:,j))];
end

Br_mtx = diag(Br_mtx_diag_elem);
Bd_mtx = diag(Bd_mtx_diag_elem);

B_mtx = blkdiag(Br_mtx,Bd_mtx);

W_mtx = pinv(B_mtx*((c0+alpha*y(2*k+3))^2*Qtau)*B_mtx');

Psi_mtx = [A_mtx'*W_mtx*A_mtx, -A_mtx'*W_mtx*b_vec; -b_vec'*W_mtx*A_mtx, b_vec'*W_mtx*b_vec];

cvx_begin

variables y(2*k+6) Y(2*k+6, 2*k+6)

expression obj

obj = trace([Y,y;y',1]*Psi_mtx);

minimize obj

subject to

y(2*k+1) == trace(Y(1:k,(k+1):2*k));

y(2*k+2) == trace(Y((k+1):2*k,(k+1):2*k));

y(2*k+4) == Y(2*k+3,2*k+3);

Y(2*k+5,2*k+5) == trace(Y(1:k,1:k)) - 2*y(2*k+1) + y(2*k+2);

y(2*k+6) == Y(2*k+3,2*k+5);

Y(2*k+3,2*k+6) == Y(2*k+4,2*k+5);

norm(y(1:k) - y((k+1):2*k)) <= y(2*k+5);

[Y,y;y',1] == semidefinite(2*k+7);

cvx_end

y_tilde = y;

end

