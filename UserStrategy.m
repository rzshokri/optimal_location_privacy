% Reza Shokri
%
% @inproceedings{Shokri2012PLP,
%  author = {Shokri, Reza and Theodorakopoulos, George and Troncoso, Carmela and Hubaux, Jean-Pierre and Le Boudec, Jean-Yves},
%  title = {Protecting location privacy: optimal strategy against localization attacks},
%  booktitle = {Proceedings of the ACM conference on computer and communications security},
%  year = {2012},
%  pages = {617--627},
%  publisher = {ACM},
% } 

% vector Prior: prior distribution (size: n) (n: number of locations)
% matrix Dp: privacy distance (size: n * n)
% matrix Dq: quality distance (size: n * m) (m: number of pseudo-locations)
% Qmax: maximum quality threshold

% OptPrivacy: privacy at the optimal-optimal point
% ObfuscationMechanism: optimal user obfuscation mechanism

function [OptPrivacy, ObfuscationMechanism] = UserStrategy(Prior, Dp, Dq, Qmax)

    pi = Prior;
    a = Dp;
    d = Dq;
    dm = Qmax;

    [k, i] = size(d);

    sq = 1;

    % Linear objective function; coefficient vector
    % the first i elements: vector u, the last k*i elements: matrix x
    xf = -1*[ones(1,i),zeros(1,k*i)]';

    % Matrix for linear inequality constraints
    xAineq = zeros(i*k+1, length(xf));

    for cj = 1:k
        w = a(cj,:).*pi;

        for ci = 1:i
            xAineq((cj-1)*i+ci, ci) = 1;
            for ck = 1:k
                xAineq((cj-1)*i+ci, i+(ck-1)*i+ci) = -w(ck);
            end
        end
    end

    for ck = 1:k
        xAineq(i*k+1, i+(ck-1)*i+1:i+ck*i) = sq * pi(ck) * d(ck, :); % service quality constraint    
    end

    % Vector for linear inequality constraints
    xbineq = [zeros(1, i*k), dm]';

    % Matrix for linear equality constraints
    xAeq = zeros(k, length(xf));

    for ck = 1:k
        xAeq(ck, i+(ck-1)*i+1:i+ck*i) = 1;
    end

    % Vector for linear equality constraints
    xbeq = ones(k, 1);

    % Vector of lower bounds
    xlb = zeros(length(xf),1);

    % Solve the LP
    [x,xfval,~,~,~] = linprog(xf,xAineq,xbineq,xAeq,xbeq,xlb);

    % The result
    xopt = ones(k,i);

    for ck = 1:k
        xopt(ck, :) = x(i+(ck-1)*i+1:i+ck*i);
    end

    OptPrivacy = - xf' * x;

    sql_opt = sum(sum(xopt .* d, 2).*pi'); %service quality loss at the optimal point

    % the obfuscation function f or Pr(r' | r)
    ObfuscationMechanism = xopt;

end
