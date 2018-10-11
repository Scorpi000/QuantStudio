% 生成 Yalmip 形式的约束条件
Constraints = [];
if exist('LinearIn_Constraint','var')
    Constraints = Constraints + [LinearIn_Constraint.A*x<=LinearIn_Constraint.b];
end
if exist('LinearEq_Constraint','var')
    Constraints = Constraints + [LinearEq_Constraint.Aeq*x==LinearEq_Constraint.beq];
end
if exist('Box_Constraint','var')
    Constraints = Constraints + [x>=Box_Constraint.lb,x<=Box_Constraint.ub];
end
if exist('Quadratic_Constraint','var')
    for i=1:length(Quadratic_Constraint)
        iQuadratic_Constraint = Quadratic_Constraint{i};
        if isfield(iQuadratic_Constraint,'X')
            Constraints = Constraints + [x'*(iQuadratic_Constraint.X*iQuadratic_Constraint.F*iQuadratic_Constraint.X'+diag(iQuadratic_Constraint.Delta))*x+iQuadratic_Constraint.Mu'*x<= iQuadratic_Constraint.q];
        else
            Constraints = Constraints + [x'*iQuadratic_Constraint.Sigma*x+iQuadratic_Constraint.Mu'*x<= iQuadratic_Constraint.q];
        end
    end
end
if exist('L1_Constraint','var')
    for i=1:length(L1_Constraint)
        iL1_Constraint = L1_Constraint{i};
        Constraints = Constraints + [norm(x-iL1_Constraint.c,1) <= iL1_Constraint.l];
    end
end
if exist('Pos_Constraint','var')
    for i=1:length(Pos_Constraint)
        iPos_Constraint = Pos_Constraint{i};
        Constraints = Constraints + [sum(abs(x-iPos_Constraint.c_pos) + (x-iPos_Constraint.c_pos)) <= 2*iPos_Constraint.l_pos];
    end
end
if exist('Neg_Constraint','var')
    for i=1:length(Neg_Constraint)
        iNeg_Constraint = Neg_Constraint{i};
        Constraints = Constraints + [sum(abs(x-iNeg_Constraint.c_neg) - (x-iNeg_Constraint.c_neg)) <= 2*iNeg_Constraint.l_neg];
    end
end
if exist('NonZeroNum_Constraint','var')
    for i=1:length(NonZeroNum_Constraint)
        iNonZeroNum_Constraint = NonZeroNum_Constraint{i};
        Constraints = Constraints + [nnz(x-iNonZeroNum_Constraint.b)<=iNonZeroNum_Constraint.N];
    end
end