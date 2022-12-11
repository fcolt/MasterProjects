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
negate(not(A),A).
negate(A,not(A)).

kb_dot_p(KB, P, Result) :- 
    findall(X, (member(X, KB), not(member(P, X))), NewKB), %remove all clauses from KB that contain P
    negate(P, NonP),                                       %to avoid non(non(p)) not reducing to p
    delete_parameter(NewKB, NonP, Result), !.

empty([]).

delete_parameter(KB, P, Result) :-
    maplist(delete_occurences_of_p(P), KB, Result). %filter the whole KB using the delete_occurences_of_p predicate

delete_occurences_of_p(P, List, Result):-
    include(non_equal_to_p(P), List, Result). %filter the current clause from kb to remove occurences of P 

non_equal_to_p(P,X) :- P \== X.

select_shortest_p(KB, P) :-
    findall(Length, (member(List, KB), length(List, Length)), Lengths),
    min_list(Lengths, MinLength),
    member(List, KB),
    length(List, MinLength),
    member(P, List).

dp_shortest_clauses_p([], []).
dp_shortest_clauses_p(KB, _) :- member([], KB), write('NO'), !, fail.
dp_shortest_clauses_p(KB, [(C/A)|S]):- 
    select_shortest_p(KB, P),
    negate(P, NonP),
    kb_dot_p(KB, P, KB_dot_P),
    dp_shortest_clauses_p(KB_dot_P, _) -> 
        write(P), write('/'), write('true'), nl, !
    ;
        kb_dot_p(KB, NonP, KB_dot_nP),
        dp_shortest_clauses_p(KB_dot_nP, _).
  

% KB=[[p,q], [not(p), a, b], [not(p), c], [d,e]],


% read_clauses_from_file(Str, []) :-
%     at_end_of_stream(Str), !.

% read_clauses_from_file(Str, [_|T]) :-
%     not(at_end_of_stream(Str)),
%     read(Str, X),
%     dp_shortest_clauses_p(X, Test) -> 
%         write(unsat), nl,
%         read_clauses_from_file(Str, T)
%     ;
%         write(sat), nl,
%     read_clauses_from_file(Str, T).


% main :-
%     open('/Users/chocogo/Desktop/Master/Projects/KRR Project 1/data_dp.in', read, Str),
%     read_clauses_from_file(Str, _),
%     close(Str).