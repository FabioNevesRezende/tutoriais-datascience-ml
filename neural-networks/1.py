# exemplo de 3 neurons recebendo 4 inputs 

inputs = [1, 2, 3, 2.5] # 4 outputs from previous layer 

weights1 = [0.2,0.8,-0.5,1] # pesos do neuronio 1
weights2 = [0.5,-0.91,0.26,-0.5] # pesos do neuronio 2
weights3 = [-0.26,-0.27,0.17,0.87] # pesos do neuronio 3

bias1 = 2 # neuronio 1
bias2 = 3 # neuronio 2
bias3 = 0.5 # neuronio 3

# como é calculado o output do neuronio: soma(input(i)*peso(i)) + viés neuronio 
output = [inputs[0]* weights1[0] + inputs[1]* weights1[1] + inputs[2]* weights1[2]+ inputs[3]* weights1[3] + bias1,
          inputs[0]* weights2[0] + inputs[1]* weights2[1] + inputs[2]* weights2[2] + inputs[3]* weights2[3] + bias2,  
          inputs[0]* weights3[0] + inputs[1]* weights3[1] + inputs[2]* weights3[2] + inputs[3]* weights3[3] + bias3
]


print(output)