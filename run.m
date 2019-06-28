clear;
load('LISWET1.mat');

unbounded_constraints = find(l <= -1e+20 & u >= 1e+20);
m = m - length(unbounded_constraints);
A(unbounded_constraints,:) = [];
l(unbounded_constraints,:) = [];
u(unbounded_constraints,:) = [];
assert(all(l == 0));
assert(all(u == 1e+20));

fraction_to_boundary = 0.99;
reduction_factor = 0.1;

x = zeros(n, 1);
mults = ones(m, 1);
slacks = ones(m, 1);
stepsize = 0.0;
for iter = 0:9999
    objective = 0.5*x'*P*x + q'*x + r;
    residual_dual = P*x + q - A'*mults;
    residual_primal = -A*x + slacks;
    avg_compl_product = dot(slacks, mults) / m;

    if mod(iter, 20) == 0
        fprintf(['iter   objective        dual feas   prim feas   ' ...
                 'min compl   avg compl   max compl   stepsize\n']);
    end
    fprintf('%4i   %.8e   %.3e   %.3e   %.3e   %.3e   %.3e   %.3e\n', ...
            iter, objective, ...
            max(abs(residual_dual)), max(abs(residual_primal)), ...
            min(slacks.*mults), avg_compl_product, max(slacks.*mults), ...
            stepsize);
    handles = plot_iterates(mults, slacks);
    if (max(abs(residual_dual)) < 1e-9 && ...
        max(abs(residual_primal)) < 1e-15 && ...
        avg_compl_product < 1e-50)
        break;
    end

    K = [           P,                      -A',             sparse(n,m);
                   -A,             sparse(m, m),                speye(m);
         sparse(m, n), spdiags(slacks, 0, m, m), spdiags(mults, 0, m, m)];
    step = K \ [-residual_dual;
                -residual_primal;
                -mults .* slacks + reduction_factor * avg_compl_product];
    [step_x, step_mults, step_slacks] = split_step(step, n, m);
    plot_steps(mults, slacks, step_mults, step_slacks, handles);

    inverse_stepsize = max([step_mults ./ (-fraction_to_boundary * mults);
                            step_slacks ./ (-fraction_to_boundary * slacks)]);
    stepsize = min(1.0, 1.0 / inverse_stepsize);
    x = x + stepsize * step_x;
    mults = mults + stepsize * step_mults;
    slacks = slacks + stepsize * step_slacks;
end



function [step_x, step_mults, step_slacks] = split_step(step, n, m)
    step_x = step(1:n);
    step_mults = step(n+1:n+m);
    step_slacks = step(n+m+1:end);
end

function [h] = plot_iterates(mults, slacks)
    color_product = [0, 0.4453125, 0.73828125];
    color_mults = [0.84765625, 0.32421875, 0.09765625];
    color_slacks = [0.92578125, 0.69140625, 0.125];

    hold off
    h(1) = semilogy(slacks.*mults, 'x', 'Color', color_product);
    hold on
    h(2) = semilogy(mults, 'x', 'Color', color_mults);
    h(3) = semilogy(slacks, 'x', 'Color', color_slacks);

    persistent all_time_ylim;
    if isempty(all_time_ylim)
        all_time_ylim = [10^-1, 10^1];
    end
    all_time_ylim(1) = min([all_time_ylim(1);
                            mults.*slacks; mults; slacks]);
    all_time_ylim(2) = max([all_time_ylim(2);
                            mults.*slacks; mults; slacks]);
    ylim([0.05 * all_time_ylim(1), 20 * all_time_ylim(2)]);
end

function plot_steps(mults, slacks, step_mults, step_slacks, handles)
    color_lambda = [0.84765625, 0.32421875, 0.09765625];
    color_s = [0.92578125, 0.69140625, 0.125];

    min_drawn = ylim;
    min_drawn = min_drawn(1);
    hold on
    semilogy(max(min_drawn, mults + step_mults), '.', ...
             'Color', color_lambda.^0.4);
    semilogy(max(min_drawn, slacks + step_slacks), '.', ...
             'Color', color_s.^0.25);
    uistack(handles, 'top');
    drawnow;
end
