#ficheiro de teste, passando dados que desejamos classificar
from pickle import load

#abrir o modelo do disco
loaded_model = load(open('binary_class_model.sav','rb'))

#tela de boas vindas
ok=True
while ok:

    print('-----------------Bem Vindo ao Sistema de Diagnostico de malaria-----------------------')
    print("")
    print("1 - Diagnosticar")
    print("0 - Sair")
    menu=int(input("Digite a opção desejada: "))
    if menu==0:
        ok=False
    if menu==1:
        #informar os sintomas ao sistema
        febre = float(input("Digite a temperatura(em graus)\n"))
        dorDeCabeca=int(input("Tem dor de cabeca? (1 - Sim 0 - Nao)\n"))
        calafrios=int(input("Tem calafrios? (1 - Sim 0 - Nao)\n"))
        suor=int(input("Está a suar de forma excessiva? (1 - Sim 0 - Nao)\n"))
        doresMusculares=int(input("Tem dores musculares? (1 - Sim 0 - Nao)\n"))
        nauseas=int(input("Tem nauseas? (1 - Sim 0 - Nao)\n"))
        dorNasCostas=int(input("Tem dor nas costas? (1 - Sim 0 - Nao)\n"))
        vomitos=int(input("Tem vomitado? (1 - Sim 0 - Nao)\n"))
        tosse=int(input("Tem tosse forte? (1 - Sim 0 - Nao)\n"))
        diarreia=int(input("Tem disenteria? (1 - Sim 0 - Nao)\n"))

        #determinar os resultados com base dataset's
        result = loaded_model.predict([
            [febre, dorDeCabeca, calafrios,suor,doresMusculares,nauseas,dorNasCostas,vomitos,tosse,diarreia]
        ])
        #mostrar o resultado, se tem malaria ou nao
        print("RESULTADO: \nTEM MALARIA" if result==1 else "RESULTADO: \nNÃO TEM MALARIA")

