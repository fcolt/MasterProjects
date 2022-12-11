%choose two clauses from KB, apply Resolution, add the Resolvent (if new to KB) => 
%new KB..., res(newKB)

%1. RES(KB) :- member(X, KB), member(Y, KB)
%2. select 2 lists from the KB (the second one should contain a negated subclause from the first list),
%(the two lists (clauses) need to be two different elements)
%3. check whether they have the connective element that allows us to apply resolution
%(like "not x or x")
%4. add the new element to the KB and delete the clauses that were used to obtain it
%(if it's an empty list ommit it)
%5. continue recursively

%returns false if satisfiable
%returns true if unsatisfiable

% my example KB = [[not(a), b], [a], [c], [d], [e], [not(e), f], [not(b), not(d), not(f)]]

%I. KB = [[not(a), b], [c,d], [not(d), b], [not(b)], [not(c), b], [e], [a, b, not(f), f]]
%II. KB = [[not(b),a], [not(a),b,e], [a, not(e)], [not(a)], [e]]
%III. KB = [[not(a),b], [c,f], [not(c)], [not(f),b], [not(c),b]]
%IV. KB = [[a,b], [not(a), not(b)], [c]]

res(KB) :- member([], KB), !.
res(KB) :- 
    sort(KB, KB_Sorted),                               %remove repeating clauses
    member(Clause1, KB_Sorted),
    delete(KB_Sorted, Clause1, KB_Without_Clause1),    %delete the first member into a temp KB
    member(Clause2, KB_Without_Clause1),               %different from the first member
    select(Subclause, Clause1, Prop1),                 
    (
        memberchk(not(Subclause), Clause2), true ->
            select(not(Subclause), Clause2, Prop2)     %if the connective element exists, select it
        ;
            false                                      %otherwise return false (satisfiable)
    ),
    append(Prop1, Prop2, Resolvent0),                  %Resolvent element
    sort(Resolvent0, Resolvent),
    delete(KB_Sorted, Clause1, KB_New1),               
    delete(KB_New1, Clause2, KB_New2),                 %delete the elements used to obtain the resolvent from the KB
    (
        not(member(_, Resolvent)), true ->             %if the resolvent is empty, return true (unsatisfiable)
            true, !
        ;
            append(KB_New2, [Resolvent], KB_New3),     %otherwise append the resolvent to the KB and continue
            res(KB_New3), !
    ).

read_clauses_from_file(Str, []) :-
    at_end_of_stream(Str), !.

read_clauses_from_file(Str, [_|T]) :-
    not(at_end_of_stream(Str)),
    read(Str, X),
    res(X) -> 
        write(unsat), nl,
        read_clauses_from_file(Str, T)
    ;
        write(sat), nl,
    read_clauses_from_file(Str, T).


main :-
    open('/Users/chocogo/Desktop/Master/Projects/KRR Project 1/data.in', read, Str),
    read_clauses_from_file(Str, _),
    close(Str).