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
% AttackMechanism: optimal inference attack mechanism

function [OptPrivacy, AttackMechanism] = AdversaryStrategy(Prior, Dp, Dq, Qmax)

    pi = Prior;
    a = Dp;
    d = Dq;
    dm = Qmax;
    
    [k, i] = size(d);

    sq = 1;

    % Linear objective function; coefficient vector
    % the first k elements: vector v, the next i*j elements: matrix y, the last
    % element: variable z
    yf = [pi,zeros(1,i*k),sq * dm]';

    % Matrix for linear inequality constraints
    yAineq = zeros(k*i, length(yf));

    for ci = 1:i
        for ck = 1:k
            yAineq((ci-1)*k+ck, ck) = -1;
            yAineq((ci-1)*k+ck, k+(ci-1)*k+1:k+ci*k) = a(ck, :);
            yAineq((ci-1)*k+ck, length(yf)) = - sq * d(ck,ci);
        end
    end

    % Vector for linear inequality constraints
    ybineq = zeros(k*i, 1);

    % Matrix for linear equality constraints
    yAeq = zeros(i, length(yf));

    for ci = 1:i
        yAeq(ci, k+(ci-1)*k+1:k+ci*k) = 1; % probability function constraint
    end

    % Vector for linear equality constraints
    ybeq = ones(i, 1);

    % Vector of lower bounds
    ylb = zeros(length(yf),1);

    % Solve the LP
    [y,yfval,~,~,~] = linprog(yf,yAineq,ybineq,yAeq,ybeq,ylb);

    % the inference function h or Pr(\hat{r} | r')
    AttackMechanism = ones(i, k);

    for ci = 1:i
        AttackMechanism(ci, :) = y(k+(ci-1)*k+1:k+ci*k);
    end
    
    OptPrivacy = yf' * y;

end
