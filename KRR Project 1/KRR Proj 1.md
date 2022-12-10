# Knowledge Reasoning and Representation Project 1

1. Resolution:  
	Create your own KB and a Question (logically entailed from KB), expressed in natural language (as the example in the last slide of C3). You can use other examples for inspiration (and indicate the source!), but you are not allowed to copy them exactly. Your KB represented in FOL must contain variables (not like the “Toddler” example in C3 page 18). The recommended size of the KB is between 4-6 sentences.

	- a)  Represent your KB in FOL, using a vocabulary that you will define.
	
	- b)  Prove “manually” (as in C3 page 29) that the Question is logically entailed from KB, by applying Resolution.

	- c)  Prove “automatically” that the Question is logically entailed from KB (by implementing Resolution in FOL)

	- d) Use your implementation of the Resolution for the propositional case, for the following sets of propositional clauses, written in CNF:
		- I.  \[\[¬a,b],\[c,d],\[¬d,b],\[ ¬b],\[¬c,b],\[e],\[a,b,¬f,f]]
		- II.  \[\[¬b,a],\[¬a,b,e],\[a, ¬e],\[ ¬a], ],\[e]]
		- III. \[\[¬a,b],\[c,f],\[ ¬c],\[¬f,b],\[¬c,b]]
		- IV.  \[\[a,b],\[ ¬a, ¬b],\[c]]

2. SAT solver – The Davis Putnam procedure  
Implement the Davis-Putnam SAT procedure. For S, a set of clauses in written in CNF, the procedure will display YES, respectively NOT, as S is satisfiable or not. In the case of YES, the procedure will also display the truth values assigned to the literals (e.g. {w/true; s/false; p/false ...}). Choose two strategies (but not in pair least/most) of selection of the atom to perform the • operation and discuss/compare the results.  
Use your implementation (with both strategies) for the following finite sets of propositional clauses, written in CNF:

	- I.  \[\[toddler],\[¬toddler,child],\[¬child,¬male,boy],\[¬infant,child], \[¬child,¬female,girl], \[female], \[girl]]
	    
	- II.  \[\[toddler],\[¬toddler,child],\[¬child,¬male,boy],\[¬infant,child], \[¬child,¬female,girl], \[female], \[¬girl]]
	    
	- III.  \[\[¬a,b],\[c,d],\[ ¬d,b],\[ ¬c,b],\[ ¬b],\[e],\[a,b,¬f,f]]
	    
	- IV.  \[\[¬b,a],\[ ¬a,b,e],\[e],\[a, ¬e],\[ ¬a]]
	    
	- V.  \[\[¬a,¬e,b],\[¬d,e,¬b],\[¬e,f,¬b],\[f,¬a,e],\[e,f,¬b]]
	    
	- VI.  \[\[a,b],\[¬a,¬b] ,\[¬a,b] ,\[a,¬b]]