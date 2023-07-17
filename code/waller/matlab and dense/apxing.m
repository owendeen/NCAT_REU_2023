%{
A = load("eris1176.mat");
A = A.Problem.A;

%}
A= load("obstclae.mat");
A = A.Problem.A;
A_pattern = spones(A);

A_pattern1 = A_pattern;
A_pattern2 = A_pattern^2;
A_pattern3 = A_pattern^3;
A_pattern4 = A_pattern^4;
A_pattern8 = A_pattern^8; 
%%
tic
guess = apx_dA_sparse(A,A_pattern1)
toc
%%
tic
guess = apx_dA_sparse(A,A_pattern2)
toc

tic
guess = apx_dA_sparse(A,A_pattern3)
toc

tic
guess = apx_dA_sparse(A,A_pattern4)
toc

%%

tic
guess = apx_dA_sparse(A,A_pattern8)
toc

%%

he = magic(4)

q = [0 , 1, 0 ,0]


hi= q*he

