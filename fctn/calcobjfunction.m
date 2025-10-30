function [ funcValue ] = calcobjfunction(para, model, X, Y )
% compose tt-tensor to full tensor
estimated_Y = contract(X,model,para.L);

funcValuePart1 = sum(reshape((Y-estimated_Y),[numel(Y),1]).^2);
% funcValuePart1 = funcValuePart1 / datasetSize;

funcValuePart2 = norm(reshape(model,[prod(para.P),prod(para.Q)]),'fro')^2;


funcValue = funcValuePart1 + para.lambda * funcValuePart2;

end

