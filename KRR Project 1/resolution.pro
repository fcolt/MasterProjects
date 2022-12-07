%choose two clauses from KB, apply Resolution, add the Resolvent (if new to KB) => 
%new KB..., res(newKB)

%1. RES(KB) :- member(X, KB), member(Y, KB)
%2. check if X and Y are not equal
%3. check whether they have the connective element that allows us to apply resolution
%(like "not x or x")
%4. add the new element to the KB
%continue recursively

% res(KB) :- member([], KB).
% res(KB) :-
%     member(X, KB),
%     member(Y, KB),

res(KB) :- member([], KB).
res(KB) :- 
    sort(KB, KB_Sorted), 
    member(Clause1, KB_Sorted),
    member(Clause2, KB_Sorted),
    select(Subclause, Clause1, Prop1),
    (
        memberchk(not(Subclause), Clause2), true ->
            select(not(Subclause), Clause2, Prop2)
        ;
            false
    ),
    append(Prop1, Prop2, Resolvent0),
    sort(Resolvent0, Resolvent),
    delete(KB_Sorted, Clause1, KB_New1),
    delete(KB_New1, Clause2, KB_New2),
    append(KB_New2, [Resolvent], KB_New3),
    res(KB_New3).


