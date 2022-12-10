% ... choose the atom C, perform the • operation for C; respectively for n(C) ... in
%both cases, for the resulting list of clauses, L1... dp(L1,S).

%for KB • p, choose a Clause L:
%1. if L contains p, remove L from KB,
%2. if L contains not(p), remove not(p) from L
% test: C = [[p,q],[not(p),a,b],[not(p),c],[d,e]]; C • p = [[q], [d,e]]

%how to choose p?
%1. p is the most balanced atom
%(the number of positive occurences is closest to the number of negatives)
%2. p appears in the shortest clause(s) in KB

%KB=[[p,q],[not(p),a,b],[not(p),c],[d,e]]
%KB=[[p,q,not(p)], [not(p))]
kb_dot_p(KB, P, Result) :- 
    findall(X, (member(X, KB), not(member(P, X))), NewKB), %remove all clauses from KB that contain P
    delete_parameter(NewKB, not(P), Result0),
    exclude(empty, Result0, Result).

empty([]).

delete_parameter(KB, P, Result):-
    maplist(delete_occurences_of_p(P), KB, Result). %filter the whole KB using the delete_occurences_of_p predicate

delete_occurences_of_p(P, List, Result):-
    include(non_equal_to_p(P), List, Result). %filter the current clause from kb to remove occurences of P 

non_equal_to_p(P,X) :- P \== X.


davis_putnam_1([], []).
davis_putnam_1(KB, _) :- member([], KB), !, fail.
davis_putnam_1(KB, [(C/A)|S]):- 
    choose_p(),
    kb_dot_p(KB, P, Result),
    empty(Result) -> 
        true,
    

