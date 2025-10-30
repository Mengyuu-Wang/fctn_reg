function [SA,phi] = CS_tensor(A,J,h,s)
    % CountSketch, A为三阶张量,tensor数据，对mode-2
    sz = size(A,1);
    phi = zeros(J,sz);
    for i= 1:sz
        phi(h(i),i) = 1;
    end
%     phi = phi * diag(s);
%     phi = (phi' .* s)';
    phi = phi .* s';
    SA = ttm(A, phi, 1);
end