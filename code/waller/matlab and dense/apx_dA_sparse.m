function apx_of_dA = apx_dA_sparse(A , pattern)



sizevec = size(A);
n = sizevec(1);

pattern = tril(pattern);
pattern = logical(pattern);


pattern = transpose(pattern);

apx_of_dA = 1;

parfor i = 1:n
    
    slicer = pattern(:,i);
    A_i = A(slicer , slicer );
    %A_i = A_i(slicer ,:);
    



    L_i = chol(A_i,'lower');
    
    apx_of_dA = apx_of_dA *(L_i( end, end) ) ^ (2/n);
end
end


%{

sizevec = size(A);
n = sizevec(1);

pattern = tril(pattern);
pattern = logical(pattern);

apx_of_dA = 1;

for i = 1:n
    slicer = pattern(i,:);
    A_i = A(:, slicer );
    A_i = A_i(slicer ,:);
    
    
    L_i = chol(A_i,'lower');
    
    apx_of_dA = apx_of_dA *(L_i( end, end) ) ^ (2/n);
%}
