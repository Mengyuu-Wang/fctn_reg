function [SA,phi] = CS_tensor(A,J,h,s)
    % CountSketch, AΪ��������,tensor���ݣ���mode-2
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