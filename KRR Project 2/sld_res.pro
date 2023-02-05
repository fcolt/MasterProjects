n(not(A), A).
n(A, not(A)).

head([H|_], H).
tail([_|T], T).

neg_list([], []).
neg_list([Head|Tail], [NegHead|NegTail]) :- 
    n(Head, NegHead),   
    neg_list(Tail, NegTail), !.

if_condition(KB, Q, QClause, KBClause_Head) :-
    QClause == KBClause_Head,
    backchaining(KB, Q).

backchaining(_, Q) :- member([], Q). 
backchaining(KB, Q) :- 
    foreach(
        member(KBClause, KB),
        do_loop(KBClause, KB, Q)
    ).

do_loop(KBClause, KB, Q) :-
    sort(KBClause, KBClause_Sorted), %sorting a positive horn clause places the positive clause first
    head(KBClause_Sorted, KBClause_Sorted_Head),
    head(Q, QClause),
    tail(KBClause_Sorted, KBClause_Sorted_Tail),
    neg_list(KBClause_Sorted_Tail, KBClause_Sorted_Tail_Pos), %from not(p1),...,not(pm) to p1,...,pm
    tail(Q, QTail),
    append(KBClause_Sorted_Tail_Pos, QTail, NewQ),
    if_condition(KB, NewQ, QClause, KBClause_Sorted_Head) -> fail; !.

read_KB_and_output_answer_bc(Str, _, []) :-
    at_end_of_stream(Str), !.

read_KB_and_output_answer_bc(Str, Q, [_|T]) :-
    not(at_end_of_stream(Str)),
    read(Str, X),
    backchaining(X, Q) -> 
        write(no), nl,
        read_KB_and_output_answer_bc(Str, Q, T)
    ;
        write(yes), nl,
        read_KB_and_output_answer_bc(Str, Q, T).

main :-
    open('data.in', read, Str),
    Questions = ['Is the person Romanian?', 'Is the person a civilian?', 'Does the person own a gun?'],
    In = [a, b, c],
    read_answers(Questions, In, Q),
    read_KB_and_output_answer_bc(Str, Q, _).

read_answers(_, [], _).
read_answers([H|T], [In_H|In_T], Out) :-
    writeln(H),
    read(Input),
    (
        Input == 'stop'
    ->
        writeln('Program stopped')
    ;
        Input == 'yes'
    ->
        append([In_H], [], Out),
        read_answers(T, In_T, Out)
    ;
        Input == 'no'
    ->
        n(In_H, In_H_Neg),
        append([In_H_Neg], [], Out),
        read_answers(T, In_T, Out)
    ).
