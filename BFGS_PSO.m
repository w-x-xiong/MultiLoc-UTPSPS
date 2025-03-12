function [y_est,fval] = BFGS_PSO(Rx, Ts, dTs, SigmaRg, SigmadRg, NPtcl, Nmax, omega_max, omega_min, c1, c2, lb, ub)
%Proposed globally optimized integrated approach

%Inputs:
%Rx - H \times L receiver position matrix
%Ts - 1 \times L indirect-path time delay measurement vector
%dTs - 1 \times L direct-path time delay measurement vector
%SigmaRg - 1 \times L std vector (for bistatic ranges)
%SigmadRg - 1 \times L std vector (for direct ranges)
%NPtcl - number of particles
%Nmax - maximum number of PSO iterations
%omega_max - upper limit for weighting factor controlling the scope of search
%omega_min - lower limit for weighting factor controlling the scope of search
%c1 - cognitive parameter
%c2 - social parameter
%lb - (2H+1) \times 1 lower bounds for optimization variables
%ub - (2H+1) \times 1 upper bounds for optimization variables

%Output:
%y_est - (2H+1) \times 1 estimate vector
%fval - objective function values

[~, L] = size(Ts);
[H, ~] = size(Rx);

fval = [];

options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton');

pjk_mtx = lb + (ub - lb).*rand(2*H+1, NPtcl);

pjb_mtx = pjk_mtx;

obj_vec = zeros(NPtcl, 1);

for n_idx = 1:NPtcl
    obj_vec(n_idx) = obj_fun(pjb_mtx(:,n_idx));
end

min_idx = find(obj_vec == min(obj_vec));

pg = pjb_mtx(:,min_idx);

PSO_idx = 0;

vjkp1_mtx = zeros(2*H+1, NPtcl);

while 1

    pj_tilde_mtx = zeros(2*H+1, NPtcl);

    for n_idx = 1:NPtcl

        if (isnan(sum(pjk_mtx(:,n_idx))) || ((sum(pjk_mtx(:,n_idx)))==0))
            pjk_mtx(:,n_idx) = lb + (ub - lb).*rand(2*H+1, 1);
        end

        %You may implement a custom quasi-Newton method tailored to your specific context
        %This is simply a demonstration of using MATLAB's built-in command fminunc
        [y_est_int,~,~,~] = fminunc(@obj_fun,pjk_mtx(:,n_idx),options);

        pj_tilde_mtx(:, n_idx) = y_est_int;

        obj_value_pjb = obj_fun(pjb_mtx(:,n_idx));

        if ((obj_fun(pj_tilde_mtx(:, n_idx)))<obj_value_pjb)
            pjb_mtx(:,n_idx) = pj_tilde_mtx(:, n_idx);
        end

    end

    for n_idx = 1:NPtcl
        obj_vec(n_idx) = obj_fun(pjb_mtx(:,n_idx));
    end

    min_idx = find(obj_vec == min(obj_vec));

    pg = pjb_mtx(:,min_idx(1));

    fval = [fval;obj_fun(pg)];

    omega_k = omega_max - ((omega_max - omega_min)/Nmax)*PSO_idx;

    vjkp1_mtx = omega_k*vjkp1_mtx + c1*rand*(pjb_mtx - pjk_mtx) + c2*rand*(pg*ones(1,NPtcl) - pjk_mtx);

    pjkp1_mtx = pjk_mtx + vjkp1_mtx;

    while ((sum(sum(pjkp1_mtx<lb)) + sum(sum(pjkp1_mtx>ub)))~=0)

        pjkp1_mtx = lb + (ub - lb).*rand(2*H+1, NPtcl);

    end

    PSO_idx = PSO_idx + 1;

    pjk_mtx = pjkp1_mtx;

    if (PSO_idx >= Nmax)
        break
    end

end

y_est = pg;

    function obj = obj_fun(y_vec)

        obj = 0;

        for l = 1:L
            obj = obj + (y_vec(2*H+1)*Ts(1,l) - norm(y_vec(1:H) - y_vec(H+1:2*H)) - norm(y_vec(1:H) - Rx(:,l)))^2/SigmaRg(1,l)^2 ...
                + (y_vec(2*H+1)*dTs(1,l) - norm(y_vec(H+1:2*H) - Rx(:,l)))^2/SigmadRg(1,l)^2;
        end

    end

end

