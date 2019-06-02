import numpy as np

iterations = 100000
count_a_failing = 0
count_none_failing = 0

# simulate joint probability A = S = 0, B = C = D = 1
for i in range(iterations):
    
    subsystem_a = np.random.binomial(1, [2/3.], size=1)
    subsystem_b = np.random.binomial(1, [2/3.], size=1)
    subsystem_c = np.random.binomial(1, [2/3.], size=1)
    subsystem_d = np.random.binomial(1, [2/3.], size=1)
    
    if subsystem_a != 1 & subsystem_b == 1 & subsystem_c == 1 & subsystem_d == 1:
        count_a_failing += 1

value_a = (count_a_failing / iterations) # joint probability A = S = 0, B = C = D = 1

# simulate probability of S = 1
for i in range(iterations):
    
    subsystem_a = np.random.binomial(1, [2/3.], size=1)
    subsystem_b = np.random.binomial(1, [2/3.], size=1)
    subsystem_c = np.random.binomial(1, [2/3.], size=1)
    subsystem_d = np.random.binomial(1, [2/3.], size=1)
    
    if subsystem_a == 1 & subsystem_b == 1 & subsystem_c == 1 & subsystem_d == 1:
        count_none_failing += 1
        
value_b = (count_none_failing / iterations) # probability of S = 1

# calculate probability of A = 0, B = C = D = 1 given S = 0
value_c = value_a / ( 1 - value_b )
print('Probability of A = 0, B = C = D = 1 given S = 0:', value_c)