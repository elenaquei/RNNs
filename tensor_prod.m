function Av = tensor_prod(A,v)
% tensor product between a 3D tensor and a vector - mildly smart
% implementation
dims = size(A);
if length(dims)~=3
    error('not coded')
end
Av = zeros(dims(1:end-1));
if isintval(A) || isintval(v)
    Av = intval(Av);
end

for i =1:size(A,1)
    for j = 1:size(A,2)
        Av(i,j) = sum(squeeze(A(i,j,:)).*v);
    end
end

end