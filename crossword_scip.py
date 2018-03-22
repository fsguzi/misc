import pyscipopt
import numpy as np

model = pyscipopt.Model("Example")

letters = list('abcdefghijklmnopqrstuvwxyz')
letters = {x:letters.index(x) for x in letters}

words = ['cat','boy','matlab','xmark','anti','morgan','zoo']

n_dim = 8
num_word = len(words)
max_len = max([len(word) for word in words])

words_alpha = np.zeros((num_word,n_dim,26), dtype='int') 
for i in range(num_word):
    word = words[i]
    for j in range(len(word)):
        letter = word[j]
        letter_idx = letters[letter]
        words_alpha[i,j,letter_idx] = 1
        
#make decision variables
O = ['H','V']
W = [w for w in range(num_word)]
R = [r for r in range(n_dim)]
C = [c for c in range(n_dim)]
decision_variables = [[[[model.addVar(o+str(w)+str(r)+str(c), vtype="BINARY") for c in C] for r in R] for w in W]for o in O]
decision_variables = np.array(decision_variables)

#make contribution matrix

contrib_matrix_horizontal = np.zeros((num_word,n_dim,n_dim,n_dim,n_dim,26), dtype='int')
#horizontal
for w in range(num_word):
    for r_d in range(n_dim):
        for c_d in range(n_dim):
            accesser = contrib_matrix_horizontal[w,r_d,c_d]
            accesser[r_d, c_d:] = words_alpha[w,:n_dim-c_d]


contrib_matrix_vertical = np.zeros((num_word,n_dim,n_dim,n_dim,n_dim,26), dtype='int')
#vertical
for w in range(num_word):
    for r_d in range(n_dim):
        for c_d in range(n_dim):
            accesser = contrib_matrix_vertical[w,r_d,c_d]
            accesser[c_d, r_d:] = words_alpha[w,:n_dim-r_d]

#CONSTRAINT 1
#everyword assigned a position
for exp in decision_variables.sum(axis=(0,2,3)):
    model.addCons(exp == 1)
    
#CONSTRAINT 2
#every grid assigned at most 2 letters

#decision coefficient
coeff_horizontal = contrib_matrix_horizontal * decision_variables[0,:,:,:,np.newaxis,np.newaxis,np.newaxis]
coeff_vertical = contrib_matrix_vertical * decision_variables[1,:,:,:,np.newaxis,np.newaxis,np.newaxis]

#actual constraint
for exps in coeff_horizontal.sum(axis=(0,1,2,5)):
    for exp in exps:
        model.addCons(exp <= 1)
        
for exps in coeff_vertical.sum(axis=(0,1,2,5)):
    for exp in exps:
        model.addCons(exp <= 1)
        
#CONSTRAINT 3
#every grid assigned at most 1 alphabet

for r in range(n_dim):
    for c in range(n_dim):
        for h_letter in range(26):
            v_letter_list = [x for x in range(26)]
            v_letter_list.remove(h_letter)
            for v_letter in v_letter_list:
                exp = coeff_horizontal[:,:,:,r,c,h_letter].sum() + coeff_vertical[:,:,:,r,c,v_letter].sum()
                if contrib_matrix_horizontal[:,:,:,r,c,h_letter].sum() + contrib_matrix_vertical[:,:,:,r,c,v_letter].sum() > 0:
                    model.addCons(exp <= 1)

model.setObjective(decision_variables.sum(), "maximize")
model.hideOutput() # silent mode
model.optimize()

for xxxx in decision_variables:
    for xxx in xxxx:
        for xx in xxx:
            for x in xx:
                if model.getVal(x) == 1:
                    print(x)
