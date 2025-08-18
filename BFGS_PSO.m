function [y_est,fval] = BFGS_PSO(Rx, Ts, dTs, SigmaRg, SigmadRg, NPtcl, Nmax, omega_max, omega_min, c1, c2, lb, ub)
% Proposed globally optimized integrated approach
% with Reflecting Boundary Handling

[~, L] = size(Ts);
[H, ~] = size(Rx);

fval = [];

options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton');

% ---- Initialization ----
pjk_mtx = lb + (ub - lb).*rand(2*H+1, NPtcl);  % initial positions
pjb_mtx = pjk_mtx;                             % personal bests
obj_vec = zeros(NPtcl, 1);

for n_idx = 1:NPtcl
    obj_vec(n_idx) = obj_fun(pjb_mtx(:,n_idx));
end

[~, min_idx] = min(obj_vec);
pg = pjb_mtx(:,min_idx);   % global best

PSO_idx = 0;
vjkp1_mtx = zeros(2*H+1, NPtcl);  % initial velocities

% ---- Main PSO loop ----
while true
    pj_tilde_mtx = zeros(2*H+1, NPtcl);

    % Local refinement with BFGS
    for n_idx = 1:NPtcl
        if (any(isnan(pjk_mtx(:,n_idx))) || all(pjk_mtx(:,n_idx)==0))
            pjk_mtx(:,n_idx) = lb + (ub - lb).*rand(2*H+1, 1);
        end

        [y_est_int,~,~,~] = fminunc(@obj_fun,pjk_mtx(:,n_idx),options);
        pj_tilde_mtx(:, n_idx) = y_est_int;

        % update personal best
        if obj_fun(pj_tilde_mtx(:, n_idx)) < obj_fun(pjb_mtx(:,n_idx))
            pjb_mtx(:,n_idx) = pj_tilde_mtx(:, n_idx);
        end
    end

    % update global best
    for n_idx = 1:NPtcl
        obj_vec(n_idx) = obj_fun(pjb_mtx(:,n_idx));
    end
    [~, min_idx] = min(obj_vec);
    pg = pjb_mtx(:,min_idx);
    fval = [fval; obj_fun(pg)];

    % inertia weight update
    omega_k = omega_max - ((omega_max - omega_min)/Nmax)*PSO_idx;

    % velocity & position update
    r1 = rand(size(pjk_mtx));
    r2 = rand(size(pjk_mtx));

    vjkp1_mtx = omega_k*vjkp1_mtx ...
        + c1*r1.*(pjb_mtx - pjk_mtx) ...
        + c2*r2.*(pg*ones(1,NPtcl) - pjk_mtx);

    pjkp1_mtx = pjk_mtx + vjkp1_mtx;

    % ---- Reflecting Boundary Handling ----
    for i = 1:(2*H+1)
        out_low  = pjkp1_mtx(i,:) < lb(i);
        out_high = pjkp1_mtx(i,:) > ub(i);

        % reflect back into domain
        pjkp1_mtx(i,out_low)  = lb(i) + (lb(i) - pjkp1_mtx(i,out_low));
        pjkp1_mtx(i,out_high) = ub(i) - (pjkp1_mtx(i,out_high) - ub(i));

        % ensure still inside bounds
        pjkp1_mtx(i,:) = max(lb(i), min(ub(i), pjkp1_mtx(i,:)));

        % reverse velocity when reflected
        vjkp1_mtx(i,out_low | out_high) = -vjkp1_mtx(i,out_low | out_high);
    end

    % iteration counter
    PSO_idx = PSO_idx + 1;
    pjk_mtx = pjkp1_mtx;

    if (PSO_idx >= Nmax)
        break
    end
end

y_est = pg;

% ---- Objective function ----
    function obj = obj_fun(y_vec)
        obj = 0;
        for l = 1:L
            obj = obj + ...
                (y_vec(2*H+1)*Ts(1,l) - norm(y_vec(1:H) - y_vec(H+1:2*H)) - norm(y_vec(1:H) - Rx(:,l)))^2 / SigmaRg(1,l)^2 ...
                + (y_vec(2*H+1)*dTs(1,l) - norm(y_vec(H+1:2*H) - Rx(:,l)))^2 / SigmadRg(1,l)^2;
        end
    end

end
