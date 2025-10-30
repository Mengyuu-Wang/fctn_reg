function newA=outer_prod(A,B)
% @author:slandarer

if length(size(B))>2||size(B,2)>1
    error('The second input should have the size of nx1.')
end
if size(A,2)==1&&length(size(A))<=2
    newSize=[size(A,1),length(B)];
else
    newSize=[size(A),length(B)];
end

newA=zeros(newSize);
orSize=numel(A);

for i=1:length(B)
    newA((1+(i-1)*orSize):(i*orSize))=A(:).*B(i);
end
end

